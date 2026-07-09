#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


HEADER = [
    "timestamp",
    "event",
    "role",
    "operation",
    "query_count",
    "total_time",
    "tag",
    "payload",
    "num_payloads",
    "num_iovs",
    "statement",
]


LOG_START_RE = re.compile(
    r"^(?:\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]\s+)?"
    r"(?P<level>\w+):\s+\[QUERY_LOG\]\s*(?P<body>.*)$"
)


def get_value(body: str, key: str, default: str = "none") -> str:
    """
    Extract key=value from the log body.

    Supports:
      event=query_start
      event=copy_tag_start,
      tag=...
    """
    match = re.search(rf"\b{re.escape(key)}=([^,\s]+)", body)
    if match:
        return match.group(1).strip()
    return default


def extract_statement(body: str) -> str:
    marker = "statement="
    if marker not in body:
        return "none"
    return body.split(marker, 1)[1].strip()


def normalize_sql(statement: str) -> str:
    statement = statement.replace("\r", " ")
    statement = statement.replace("\n", " ")
    statement = re.sub(r"\s+", " ", statement)
    return statement.strip() or "none"


def build_metadata(timestamp: str, body: str) -> dict:
    return {
        "timestamp": timestamp or "none",
        "event": get_value(body, "event"),
        "role": get_value(body, "role"),
        "operation": "none",
        "query_count": get_value(body, "query_count"),
        "total_time": get_value(body, "total_time"),
        "tag": get_value(body, "tag"),
        "payload": get_value(body, "payload"),
        "num_payloads": get_value(body, "num_payloads"),
        "num_iovs": get_value(body, "num_iovs"),
        "statement": "none",
    }


def current_operation(active_tag: bool, active_payload: bool) -> str:
    if active_payload:
        return "copy_payload"

    if active_tag:
        return "copy_tag"

    return "copy"


def parse_query_log(log_path: Path) -> list[dict]:
    rows = []

    pending_query = None
    in_statement = False

    active_tag = False
    active_payload = False

    current_tag = "none"
    current_payload = "none"

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            match = LOG_START_RE.match(line)

            if match:
                timestamp = match.group("timestamp") or "none"
                body = match.group("body")

                metadata = build_metadata(timestamp, body)
                event = metadata["event"]

                # -----------------------------
                # State-changing events
                # -----------------------------

                if event == "copy_tag_start":
                    active_tag = True
                    active_payload = False

                    current_tag = metadata["tag"]
                    current_payload = "none"

                    # Do not write copy_tag_start to CSV
                    continue

                if event == "copy_payload_start":
                    active_payload = True
                    current_payload = metadata["payload"]

                    # Do not write copy_payload_start to CSV
                    continue

                if event == "copy_payload":
                    # This is the end/summary of one payload copy.
                    metadata["operation"] = "copy_payload"
                    metadata["tag"] = current_tag
                    metadata["payload"] = (
                        metadata["payload"]
                        if metadata["payload"] != "none"
                        else current_payload
                    )

                    rows.append(metadata)

                    # After payload finishes, we are back inside copy_tag
                    active_payload = False
                    current_payload = "none"
                    continue

                if event == "copy_tag":
                    # This is the end/summary of the tag copy.
                    metadata["operation"] = "copy_tag"
                    metadata["tag"] = (
                        metadata["tag"]
                        if metadata["tag"] != "none"
                        else current_tag
                    )

                    rows.append(metadata)

                    # After tag finishes, we are back at top-level copy
                    active_tag = False
                    active_payload = False
                    current_tag = "none"
                    current_payload = "none"
                    continue

                # -----------------------------
                # Query events
                # -----------------------------

                if event == "query_start":
                    if pending_query:
                        pending_query["statement"] = normalize_sql(
                            pending_query["statement"]
                        )
                        rows.append(pending_query)

                    pending_query = {
                        "timestamp": timestamp,
                        "event": "query",
                        "role": metadata["role"],
                        "operation": current_operation(active_tag, active_payload),
                        "query_count": "none",
                        "total_time": "none",
                        "tag": current_tag if current_tag != "none" else "none",
                        "payload": current_payload if current_payload != "none" else "none",
                        "num_payloads": "none",
                        "num_iovs": "none",
                        "statement": extract_statement(body),
                    }

                    in_statement = True
                    continue

                if event == "query_complete":
                    if pending_query:
                        pending_query["total_time"] = metadata["total_time"]
                        pending_query["statement"] = normalize_sql(
                            pending_query["statement"]
                        )
                        rows.append(pending_query)

                        pending_query = None
                        in_statement = False
                    else:
                        metadata["event"] = "query"
                        metadata["operation"] = current_operation(
                            active_tag,
                            active_payload,
                        )
                        metadata["tag"] = current_tag
                        metadata["payload"] = current_payload
                        rows.append(metadata)

                    continue

                # -----------------------------
                # Other QUERY_LOG events
                # -----------------------------

                if pending_query:
                    pending_query["statement"] = normalize_sql(
                        pending_query["statement"]
                    )
                    rows.append(pending_query)
                    pending_query = None
                    in_statement = False

                metadata["operation"] = current_operation(active_tag, active_payload)

                if metadata["tag"] == "none":
                    metadata["tag"] = current_tag

                if metadata["payload"] == "none":
                    metadata["payload"] = current_payload

                rows.append(metadata)
                continue

            # Continuation line, probably part of a multiline SQL statement
            if in_statement and pending_query:
                pending_query["statement"] += " " + line.strip()

    if pending_query:
        pending_query["statement"] = normalize_sql(pending_query["statement"])
        rows.append(pending_query)

    return rows


def write_csv(rows: list[dict], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse conddb QUERY_LOG lines into one-row-per-query CSV."
    )
    parser.add_argument("logfile", help="Path to the .log file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Defaults to <logfile>_query_log.csv",
    )

    args = parser.parse_args()

    log_path = Path(args.logfile)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    if args.output:
        csv_path = Path(args.output)
    else:
        csv_path = log_path.with_name(log_path.stem + "_query_log.csv")

    rows = parse_query_log(log_path)
    write_csv(rows, csv_path)

    print(f"Parsed {len(rows)} QUERY_LOG row(s)")
    print(f"Wrote CSV file: {csv_path}")


if __name__ == "__main__":
    main()