# average_dummy_results.py

import os
import sys
import json
import re
import pandas as pd
from collections import defaultdict

# === CONFIGURATION ===

if len(sys.argv) != 2:
    print("Usage: python average_dummy_results.py <path_to_results>")
    sys.exit(1)

base_dir = sys.argv[1]

prefixes = ["local_", "remote_", "whole_"]

# === FUNCTIONS ===

def find_all_json_files(base_dir, prefix):
    grouped_entries = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        rel_root = os.path.relpath(root, base_dir)

        if rel_root == ".":
            continue  # Skip base directory itself

        folder = os.path.basename(root)
        print(folder)
        folder_match = re.match(r'.*_m(\d+)_t(\d+)s(\d+)', folder)
        if not folder_match:
            print("skip")
            continue  # skip non-test folders

        message_size, threads, streams = map(int, folder_match.groups())

        for fname in sorted(files):
            print(fname)
            if fname.endswith(".json") and fname.startswith(prefix):
                full_path = os.path.join(root, fname)

                try:
                    with open(full_path) as f:
                        data = json.load(f)

                    events = data.get("total", {}).get("events", 1)
                    recv_modules = [m for m in data["modules"] if m.get("type") == "MPIReceiver"]
                    send_modules = [m for m in data["modules"] if m.get("type") == "MPISender"]

                    entry = {}

                    if recv_modules:
                        entry["recv_cpu"] = sum(m.get("time_thread", 0.0) for m in recv_modules) / events
                        entry["recv_real"] = sum(m.get("time_real", 0.0) for m in recv_modules) / events

                    if send_modules:
                        entry["send_cpu"] = sum(m.get("time_thread", 0.0) for m in send_modules) / events
                        entry["send_real"] = sum(m.get("time_real", 0.0) for m in send_modules) / events

                    entry["total_cpu"] = data.get("total", {}).get("time_thread", 0.0) / events
                    entry["total_real"] = data.get("total", {}).get("time_real", 0.0) / events

                    key = (message_size, threads, streams)
                    grouped_entries[key].append(entry)

                except Exception as e:
                    print(f"Error in {full_path}: {e}")
                    continue

    return grouped_entries

def parse_throughput_file(filepath):
    throughput_data = defaultdict(list)
    pattern = re.compile(
        r"\[THROUGHPUT\] .*_m(\d+)_t(\d+)s(\d+)_r(\d+) \| avg: ([\d.]+)"
    )

    with open(filepath) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                message_size, threads, streams, run_id, avg_val = match.groups()
                key = (int(message_size), int(threads), int(streams))
                throughput_data[key].append((int(run_id), float(avg_val)))

    # Discard warmup (first run by run_id) and average the rest
    throughput_avg = {}
    for key, values in throughput_data.items():
        values.sort()  # sort by run_id
        valid = values[1:]  # discard first
        if valid:
            avg_throughput = sum(v[1] for v in valid) / len(valid)
            throughput_avg[key] = avg_throughput

    return throughput_avg

def summarize(grouped_entries):
    summary = []
    for (msg_sz, th, st), group in grouped_entries.items():
        if len(group) <= 1:
            continue  # not enough runs

        df = pd.DataFrame(group[1:])  # discard warmup
        avg = df.mean(numeric_only=True).to_dict()
        avg.update({
            "message_size_bytes": msg_sz,
            "threads": th,
            "streams": st
        })
        summary.append(avg)

    return pd.DataFrame(summary)

def attach_throughput(summary_df, throughput_avg):
    summary_df["throughput_ev_per_s"] = summary_df.apply(
        lambda row: throughput_avg.get(
            (row["message_size_bytes"], row["threads"], row["streams"]), None
        ),
        axis=1
    )
    return summary_df

# === MAIN ===

for prefix in prefixes:
    print(f"\nProcessing prefix: {prefix}")

    grouped = find_all_json_files(base_dir, prefix)
    if not grouped:
        print(f"No data found for prefix {prefix}, skipping.")
        continue

    summary_df = summarize(grouped)
    summary_df.sort_values(by=["message_size_bytes", "threads", "streams"], inplace=True)

    throughput_path = os.path.join(base_dir, "throughputs.txt")
    throughput_avg = parse_throughput_file(throughput_path)
    summary_df = attach_throughput(summary_df, throughput_avg)

    # Save output
    output_csv = os.path.join(base_dir, f"{prefix.strip('_')}_summary_table.csv")
    output_json = os.path.join(base_dir, f"{prefix.strip('_')}_summary_table.json")

    summary_df.to_csv(output_csv, index=False)
    summary_df.to_json(output_json, orient="records", indent=2)

    print(f"✅ Saved summary to:\n- {output_csv}\n- {output_json}")

print("\n🎉 All summaries processed successfully!")

