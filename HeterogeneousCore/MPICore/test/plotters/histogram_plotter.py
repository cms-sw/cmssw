import pandas as pd
import matplotlib.pyplot as plt
import os

# ==== INPUT FILES ====

files = [
    # {"Single process (reference)": "/data/user/apolova/ompi5_test/CMSSW_15_1_0_pre5/src/HeterogeneousCore/MPICore/test/test_results/whole_hlt_t-s-c/whole_summary_table.csv"},
    {"With plugins": "/data/user/apolova/CMSSW_16_0_0_pre1/src/HeterogeneousCore/MPICore/test/test_results/ompi_cmssw_16/with_plugins/local_summary_table.csv"},
    {"Without plugins": "/data/user/apolova/CMSSW_16_0_0_pre1/src/HeterogeneousCore/MPICore/test/test_results/ompi_cmssw_16/without_plugins/local_summary_table.csv"}


]

output_dir = "./comp_plugins"

# ==== READ & PREPARE DATA ====

dfs = []

for file_info in files:
    type_name, path = next(iter(file_info.items()))
    try:
        df = pd.read_csv(path)
        df["type"] = type_name
        dfs.append(df)
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")

# Combine all available DataFrames
if not dfs:
    print("No valid files found. Exiting.")
    exit(1)

df = pd.concat(dfs)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ==== PLOTTING FUNCTIONS ====

import numpy as np

def make_histogram_throughput(df, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    threads = sorted(df['threads'].unique())

    # Separate reference vs. MPI approaches
    ref_df = df[df['type'] == "Single process (reference)"].set_index('threads')
    mpi_df = df[df['type'] != "Single process (reference)"]

    types = mpi_df['type'].unique()
    bar_width = 0.12
    num_types = len(types)
    x = np.arange(len(threads))  # Base x locations for thread groups

    plt.figure(figsize=(7, 7))
    plt.grid(axis='y', linestyle='--')

    colors = ["#336971", "#cb932a", "#a31621", "#758f00"]

    # --- Plot histogram bars for MPI approaches ---
    for i, t in enumerate(types):
        sub = mpi_df[mpi_df['type'] == t].set_index('threads')

        heights = []
        errors = []
        for th in threads:
            if th in sub.index:
                row = sub.loc[th]
                heights.append(row['throughput_ev_per_s'])
                errors.append(
                    row['throughput_error']
                    if 'throughput_error' in row and not pd.isna(row['throughput_error'])
                    else 0
                )
            else:
                heights.append(0)
                errors.append(0)

        x_pos = x + i * bar_width
        if any(errors):
            plt.bar(x_pos, heights, bar_width, label=t, yerr=errors, capsize=3, color=colors[i % len(colors)])
        else:
            plt.bar(x_pos, heights, bar_width, label=t, color=colors[i % len(colors)])

    # # --- Plot reference as horizontal dashed lines for each thread ---
    # for idx, th in enumerate(threads):
    #     if th in ref_df.index:
    #         ref_val = ref_df.loc[th]['throughput_ev_per_s']
    #         group_center = x[idx] + (num_types - 1) * bar_width / 2
    #         # horizontal line across the width of the group
    #         plt.hlines(
    #             y=ref_val,
    #             xmin=group_center - bar_width * (num_types + 2) / 2,
    #             xmax=group_center + bar_width * (num_types + 2) / 2,
    #             colors="black",
    #             linestyles="--",
    #             linewidth=2,
    #         )

    # # Add dummy legend entry for reference line
    # plt.plot([], [], linestyle="--", color="black", linewidth=2, label="Single process (reference)")

    # --- Axis labels & legend ---
    plt.xticks(x + (num_types - 1) * bar_width / 2, threads, fontsize='large')
    plt.xlabel("Threads", fontsize='large')
    plt.ylabel("Throughput (events/s)", fontsize='large')
    plt.title("Throughput per Thread Count (OpenMPI on one machine)")
    plt.legend(loc='upper left', fontsize='large')
    plt.tight_layout()

    # --- Save ---
    plt.savefig(os.path.join(output_path, "throughput_histogram.png"))
    plt.close()

# ==== MAKE PLOTS ====
make_histogram_throughput(df, output_dir)



print("✅ Plots saved to:", output_dir)