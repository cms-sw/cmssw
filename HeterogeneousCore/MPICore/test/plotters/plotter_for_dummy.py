import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===

# Path to your summary CSV file
summary_file = "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/dummy/mpich/async/different_machines/remote_summary_table.csv"

# === LOAD DATA ===

df = pd.read_csv(summary_file)

# === HELPERS ===

# Helper to format message size nicely
def format_size(bytes_val):
    if bytes_val >= 1024 * 1024:
        return f"{bytes_val // (1024 * 1024)} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val // 1024} KB"
    else:
        return f"{bytes_val} B"

# === PLOT ===

plt.figure(figsize=(10, 6))

# Group by message size
for message_size, group in df.groupby("message_size_bytes"):
    group_sorted = group.sort_values("threads")  # make sure it's ordered
    plt.plot(
        group_sorted["threads"],
        group_sorted["throughput_ev_per_s"],
        marker="o",
        label=f"{format_size(message_size)}"
    )

plt.title("Throughput Scaling with Number of Threads (per Message Size)")
plt.xlabel("Number of Threads")
plt.ylabel("Throughput (events/s)")
plt.grid(True)
plt.legend(title="Message Size", loc="best")
plt.tight_layout()

plt.savefig("dummy_scaling_mpich_async_mg")
