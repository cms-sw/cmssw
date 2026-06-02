#!/usr/bin/env python3
import argparse
import json
import sys

BYTES_TO_KB = 1024.0

# Metrics expected in each eventInfo entry.
METRICS = [
    "TotalMemoryGrowth",
    "AvgRetained",
    "AvgDataProductSize",
    "AvgTempSize",
    "AvgNTemp",
]

# Byte-based metrics are converted to kB in the output.
BYTE_METRICS = [
    "TotalMemoryGrowth",
    "AvgRetained",
    "AvgDataProductSize",
    "AvgTempSize",
]

# Mapping from memoryReports average fields to Circles metric names.
REPORT_TO_METRIC = {
    "AvgTempSize": "Average temporary allocated memory",
    "AvgNTemp": "Average number of temporary allocations",
    "AvgDataProductSize": "Average allocated memoory for data products",
    "AvgRetained": "Average allocationed memory retained between events",
    "TotalMemoryGrowth": "Total allocated memory retained through the job"
}


def build_resources():
    resources = []
    for metric, description in REPORT_TO_METRIC.items():
        if metric in BYTE_METRICS:
            unit = "kB"
        else:
            unit = ""
  
        title = f"{REPORT_TO_METRIC[metric]}"
        description = f"{metric}"

        resources.append(
            {
                "name": metric,
                "description": description,
                "title": title,
                "unit": unit,
            }
        )
    return resources


def values_from_module_event_info(event_info):
    n_events = len(event_info)
    if n_events == 0:
        return {metric: 0.0 for metric in METRICS}, 0

    sums = {metric: 0.0 for metric in METRICS}
    for entry in event_info:
        for metric in METRICS:
            sums[metric] += entry.get(metric, 0)

    averages = {}
    for metric in METRICS:
        value = sums[metric] / n_events
        if metric in BYTE_METRICS:
            value /= BYTES_TO_KB
        averages[metric] = value

    return averages, n_events


def fill_from_memory_report(memory_report, values):
    """Populate known metrics from memoryReports averages when available."""
    if not isinstance(memory_report, dict):
        return
    for metric in METRICS:
        value = memory_report.get(metric, 0)
        if metric in BYTE_METRICS:
            value = float(value) / BYTES_TO_KB
        values[metric] = float(value)

def event_count_from_memory_report(memory_report):
    if not isinstance(memory_report, dict):
        return 0

    # Pick any per-event list to infer transitions count.
    for key in (
        "TempSizeEachEvent",
        "DataProductSizeEachEvent",
        "RetainedEachEvent",
        "NTempEachEvent",
        "MemoryGrowthEachEvent",
    ):
        values = memory_report.get(key)
        if isinstance(values, list):
            return len(values)
    return 0


def format_to_circles(doc):
    modules_in = doc.get('eventData').get('modules')
    if not isinstance(modules_in, list):
        raise ValueError("Missing or invalid 'modules' field in input JSON")

    memory_reports = doc.get("memoryReports")
    if not isinstance(memory_reports, dict):
        memory_reports = {}

    out = {
        "modules": [],
        "resources": build_resources(),
        "total": {"label": "Job", "type": "Job"},
    }

    for metric in METRICS:
        out["total"][metric] = 0.0

    for module in modules_in:
        module_label = module.get("label", "")
        module_type = module.get("type", "")
        memory_report = memory_reports.get('%s-%s' % (module_label, module_type),{})
        print(f'{memory_report}')
        print()
        values = dict()
        for metric in METRICS:
            values[metric] = 0.0
        n_events = 0
        if memory_report:
            fill_from_memory_report(memory_report, values)
            n_events = max(n_events, event_count_from_memory_report(memory_report))

        out_module = {
            "label": module_label,
            "type": module_type,
            "events": max(n_events, 1),
        }
        out_module.update(values)
        out["modules"].append(out_module)

        for metric in METRICS:
            out["total"][metric] += values[metric]

    return out


def main(args):
    try:
        doc = json.load(args.filename)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error reading file: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        result = format_to_circles(doc)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        try:
            json.dump(result, args.output, indent=2)
            args.output.write("\n")
        except Exception as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        json.dump(result, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert module-event allocation JSON to Circles JSON"
    )
    parser.add_argument(
        "filename",
        type=argparse.FileType("r"),
        help="Input JSON file produced by edmModuleEventAllocMonitorAnalyze.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=None,
        help="Output file for Circles JSON (defaults to stdout)",
    )
    main(parser.parse_args())
