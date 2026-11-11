import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_VIDEO = BASE_DIR / "input1.mp4"
OUTPUT_VIDEO = BASE_DIR / "output.mp4"

# Already trained YOLO11 VisDrone model
MODEL = BASE_DIR / "best.pt"

CONFIDENCE = 0.12
IOU = 0.5
IMAGE_SIZE = 1280

DEVICE = None  # None = automatic, "cpu" or 0 = GPU

USE_REID = True
USE_GMC = True

DETECTION_ONLY = False

TRACK_BUFFER = 90
TRAIL_LENGTH = 35


# ==========================================
# TRACKER CONFIGURATION
# ==========================================

def make_tracker_config(path: Path) -> None:
    defaults = Path(check_yaml("botsort.yaml"))

    with defaults.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    settings["tracker_type"] = "botsort"
    settings["gmc_method"] = "sparseOptFlow" if USE_GMC else "none"

    settings["with_reid"] = USE_REID
    settings["model"] = "auto"

    settings["track_buffer"] = TRACK_BUFFER

    settings["track_high_thresh"] = 0.25
    settings["track_low_thresh"] = 0.10
    settings["new_track_thresh"] = 0.30

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(settings, file, sort_keys=False)


# ==========================================
# VISDRONE CLASSES
# ==========================================

def get_person_classes(model: YOLO) -> list[int]:
    """
    VisDrone:
        0 = pedestrian
        1 = people

    COCO:
        0 = person

    Automatically determines correct IDs.
    """

    names = model.names

    if isinstance(names, dict):
        entries = names.items()
    else:
        entries = enumerate(names)

    person_classes = [
        int(class_id)
        for class_id, name in entries
        if str(name).strip().lower()
        in {"pedestrian", "people", "person"}
    ]

    if not person_classes:
        raise ValueError(
            f"No person classes found in model: {names}"
        )

    return person_classes


def color_for(identity: int) -> tuple[int, int, int]:
    return (
        (identity * 67) % 200 + 40,
        (identity * 97) % 200 + 40,
        (identity * 137) % 200 + 40,
    )


# ==========================================
# MAIN
# ==========================================

def main():

    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    if not MODEL.is_file():
        raise FileNotFoundError(
            f"VisDrone model not found: {MODEL}"
        )

    print(f"Loading model: {MODEL}")

    model = YOLO(str(MODEL))

    person_classes = get_person_classes(model)

    print(f"Model classes: {model.names}")
    print(f"Person class IDs: {person_classes}")

    tracker_path = BASE_DIR / "botsort_custom.yaml"

    if not DETECTION_ONLY:
        make_tracker_config(tracker_path)

    capture = cv2.VideoCapture(str(INPUT_VIDEO))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {INPUT_VIDEO}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            "Could not read video dimensions"
        )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create video: {OUTPUT_VIDEO}"
        )

    track_history = defaultdict(
        lambda: deque(maxlen=TRAIL_LENGTH)
    )

    observed_ids = set()

    frame_index = 0

    started = time.perf_counter()

    csv_path = OUTPUT_VIDEO.with_suffix(".csv")

    try:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame",
                "time_seconds",
                "track_id",
                "class_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
            ])

            while True:

                ok, frame = capture.read()

                if not ok:
                    break

                frame_index += 1

                common_args = {
                    "classes": person_classes,
                    "conf": CONFIDENCE,
                    "iou": IOU,
                    "imgsz": IMAGE_SIZE,
                    "verbose": False,
                    "device": DEVICE,
                }

                # ==================================
                # DETECTION / TRACKING
                # ==================================

                if DETECTION_ONLY:

                    result = model.predict(
                        frame,
                        **common_args,
                    )[0]

                else:

                    result = model.track(
                        frame,
                        persist=True,
                        tracker=str(
                            tracker_path.resolve()
                        ),
                        **common_args,
                    )[0]

                annotated = frame.copy()

                visible = 0

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    boxes = (
                        result.boxes.xyxy
                        .cpu()
                        .numpy()
                    )

                    confidences = (
                        result.boxes.conf
                        .cpu()
                        .numpy()
                    )

                    class_ids = (
                        result.boxes.cls
                        .int()
                        .cpu()
                        .tolist()
                    )

                    if (
                        not DETECTION_ONLY
                        and result.boxes.id is not None
                    ):

                        ids = (
                            result.boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )

                    else:
                        ids = [None] * len(boxes)

                    visible = len(boxes)

                    for box, track_id, class_id, confidence in zip(
                        boxes,
                        ids,
                        class_ids,
                        confidences,
                    ):

                        x1, y1, x2, y2 = map(
                            int, box
                        )

                        if track_id is not None:

                            observed_ids.add(track_id)

                            color = color_for(track_id)

                            label = f"ID {track_id}"

                        else:

                            color = (0, 255, 255)

                            label = (
                                f"Person {confidence:.2f}"
                            )

                        # Bounding box
                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        # Label
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                        )

                        # ==================================
                        # TRAJECTORY
                        # ==================================

                        if track_id is not None:

                            center = (
                                (x1 + x2) // 2,
                                (y1 + y2) // 2,
                            )

                            track_history[
                                track_id
                            ].append(center)

                            points = list(
                                track_history[track_id]
                            )

                            for a, b in zip(
                                points,
                                points[1:],
                            ):

                                cv2.line(
                                    annotated,
                                    a,
                                    b,
                                    color,
                                    2,
                                )

                        # ==================================
                        # CSV EXPORT
                        # ==================================

                        csv_writer.writerow([
                            frame_index,
                            round(
                                (frame_index - 1) / fps,
                                3,
                            ),
                            (
                                track_id
                                if track_id is not None
                                else ""
                            ),
                            class_id,
                            x1,
                            y1,
                            x2,
                            y2,
                            round(
                                float(confidence),
                                4,
                            ),
                        ])

                # ==================================
                # INFORMATION PANEL
                # ==================================

                panel_height = (
                    65 if DETECTION_ONLY else 105
                )

                cv2.rectangle(
                    annotated,
                    (8, 8),
                    (
                        min(width - 1, 450),
                        min(height - 1, panel_height),
                    ),
                    (20, 20, 20),
                    -1,
                )

                cv2.putText(
                    annotated,
                    f"Detected people: {visible}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                if not DETECTION_ONLY:

                    cv2.putText(
                        annotated,
                        (
                            "Observed track IDs: "
                            f"{len(observed_ids)}"
                        ),
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                writer.write(annotated)

                if frame_index % 100 == 0:

                    print(
                        f"Processed {frame_index} frames "
                        f"| visible: {visible} "
                        f"| observed IDs: "
                        f"{len(observed_ids)}"
                    )

    finally:
        capture.release()
        writer.release()

    elapsed = (
        time.perf_counter() - started
    )

    print("\n========== RESULTS ==========")

    print(f"Frames: {frame_index}")

    print(f"Processing time: {elapsed:.1f}s")

    if elapsed > 0:

        print(
            f"Processing FPS: "
            f"{frame_index / elapsed:.2f}"
        )

    print(
        f"Observed track IDs: "
        f"{len(observed_ids)}"
    )

    print(
        "Note: track IDs are not necessarily "
        "unique real-world people."
    )

    print(f"Video: {OUTPUT_VIDEO.resolve()}")
    print(f"CSV: {csv_path.resolve()}")

    print("=============================")


if __name__ == "__main__":
    main()