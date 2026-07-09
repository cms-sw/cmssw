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
    "total_time",
    "payload",
    "tag",
    "num_payloads",
    "num_iovs",
    "query_type",
    "statement",
    "campaign",
    "payload_size",
    "payload_number",
    "test_execution",
    "test_time",
    "source",
    "destination",
    "log_file",
]


LOG_START_RE = re.compile(
    r"^(?:\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]\s+)?"
    r"(?P<level>\w+):\s+\[QUERY_LOG\]\s*(?P<body>.*)$"
)

LOG_FILE_RE = re.compile(
    r"^(?P<campaign>.+)_s(?P<payload_size>\d+)_n(?P<payload_number>\d+)_t(?P<test_execution>\d+)_(?P<test_time>\d{4}-\d{2}-\d{2}-\d{2}h\d{2}m\d{2})$"
)

def parse_log_file_metadata(log_path: Path) -> dict:
    """
    Extract metadata from filenames like:

        test1_s10_n5_t2_2026-07-09-19h39m48.log

    Result:
        campaign       = test1
        payload_size   = 10
        payload_number = 5
        test_execution = 2
        test_time      = 2026-07-09-19h39m48
    """
    match = LOG_FILE_RE.match(log_path.stem)

    if not match:
        raise ValueError(
            f"Log filename does not match expected format: {log_path.name}\n"
            "Expected format: <campaign>_s<size>_n<number>_t<execution>_<YYYY-MM-DD-HHhMMmSS>.log\n"
            "Example: test1_s10_n5_t2_2026-07-09-19h39m48.log"
        )

    metadata = match.groupdict()

    return {
        "campaign": metadata["campaign"],
        "payload_size": metadata["payload_size"],
        "payload_number": metadata["payload_number"],
        "test_execution": metadata["test_execution"],
        "test_time": metadata["test_time"],
    }


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

def get_query_type(statement: str) -> str:
    """
    Extract the SQL query type from the first word of the statement.

    Examples:
      SELECT ... -> select
      INSERT ... -> insert
      UPDATE ... -> update
      DELETE ... -> delete
    """
    if not statement or statement == "none":
        return "none"

    statement = statement.strip()

    # Remove leading SQL comments if any
    statement = re.sub(r"^/\*.*?\*/\s*", "", statement)
    statement = re.sub(r"^--.*?\n\s*", "", statement)

    match = re.match(r"^([A-Za-z]+)", statement)

    if not match:
        return "none"

    return match.group(1).lower()


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
        "total_time": get_value(body, "total_time"),
        "payload": get_value(body, "payload"),
        "tag": get_value(body, "tag"),
        "num_payloads": get_value(body, "num_payloads"),
        "num_iovs": get_value(body, "num_iovs"),
        "query_type": "none",
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

                    statement = extract_statement(body)

                    pending_query = {
                        "timestamp": timestamp,
                        "event": "query",
                        "role": metadata["role"],
                        "operation": current_operation(active_tag, active_payload),
                        "total_time": "none",
                        "payload": current_payload if current_payload != "none" else "none",
                        "tag": current_tag if current_tag != "none" else "none",
                        "num_payloads": "none",
                        "num_iovs": "none",
                        "query_type": get_query_type(statement),
                        "statement": statement,
                    }

                    in_statement = True
                    continue

                if event == "query_complete":
                    if pending_query:
                        pending_query["total_time"] = metadata["total_time"]
                        pending_query["statement"] = normalize_sql(
                            pending_query["statement"]
                        )
                        pending_query["query_type"] = get_query_type(
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
                        metadata["payload"] = current_payload
                        metadata["tag"] = current_tag
                        metadata["query_type"] = get_query_type(metadata["statement"])
                        rows.append(metadata)

                    continue

                # -----------------------------
                # Other QUERY_LOG events
                # -----------------------------

                if pending_query:
                    pending_query["statement"] = normalize_sql(pending_query["statement"])
                    pending_query["query_type"] = get_query_type(pending_query["statement"])
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
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse multiple conddb QUERY_LOG files into one consolidated CSV."
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing .log files",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output consolidated CSV path",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source database label to append to each row",
    )

    parser.add_argument(
        "--destination",
        required=True,
        help="Destination database label to append to each row",
    )

    parser.add_argument(
        "--glob",
        default="*.log",
        help="Glob pattern for log files. Default: *.log",
    )

    parser.add_argument(
        "--skip-bad-filenames",
        action="store_true",
        help="Skip log files whose names do not match the expected metadata format.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    log_files = sorted(input_dir.glob(args.glob))

    if not log_files:
        raise FileNotFoundError(
            f"No log files found in {input_dir} matching pattern {args.glob}"
        )

    all_rows = []

    for log_path in log_files:
        try:
            file_metadata = parse_log_file_metadata(log_path)
        except ValueError as error:
            if args.skip_bad_filenames:
                print(f"Skipping {log_path.name}: {error}")
                continue
            raise

        rows = parse_query_log(log_path)

        for row in rows:
            row.update(file_metadata)
            row["source"] = args.source
            row["destination"] = args.destination
            row["log_file"] = log_path.name

        all_rows.extend(rows)

        print(f"Parsed {len(rows)} QUERY_LOG row(s) from {log_path.name}")

    write_csv(all_rows, csv_path)

    print(f"Parsed {len(all_rows)} total QUERY_LOG row(s)")
    print(f"Wrote consolidated CSV file: {csv_path}")


if __name__ == "__main__":
    main()