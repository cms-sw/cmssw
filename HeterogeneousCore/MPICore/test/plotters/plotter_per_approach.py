import pandas as pd
import matplotlib.pyplot as plt
import os

# ==== INPUT FILES ====

files = [
    {"demo": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/sync/milan-genoa_ucx_t-s-c/local_summary_table.csv"},
    # {"async": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/simple_async/milan-genoa_ucx_t-s-c/local_summary_table.csv"},
    # {"one-sided": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/one-sided/milan-genoa_ucx_t-s-c/local_summary_table.csv"},
    {"whole": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/whole_hlt_t-s-c/whole_summary_table.csv"},
    {"synchronous mpich": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/mpich/sync/milan-genoa_ucx_t-s-c/local_summary_table.csv"},
    {"simple asynchronous mpich": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/mpich/simple_async/milan-genoa_ucx_t-s-c/local_summary_table.csv"},
    # {"number of products mpich": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/mpich/async_number_of_products/local-remote_t-s-c_different-sockets/local_summary_table.csv"},
    {"mpich async number of products": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/mpich/async_number_of_products/milan-genoa_ucx_t-s-c/local_summary_table.csv"},
    {"number of products with ssend": "/data/user/apolova/dev1/CMSSW_15_0_0/src/HeterogeneousCore/MPICore/test/test_results_thesis/mpich/async_number_of_products_ssend/milan-genoa_ucx_t-s-c/local_summary_table.csv"}

]

output_dir = "./milan-genoa-comparative_plots_diff_approaches"

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

def make_plot(y_column, ylabel, title, filename, max_threads=None):
    plt.figure(figsize=(10, 8))  # Slightly taller to fit legend
    for t in df['type'].unique():
        sub = df[df['type'] == t]
        if max_threads is not None:
            sub = sub[sub['threads'] <= max_threads]
        plt.plot(sub['threads'], sub[y_column], marker='o', label=t)

    plt.xlabel("Threads")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    # Move legend below
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize='small', frameon=False)

    # Adjust layout so everything fits
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()


# ==== MAKE PLOTS ====

make_plot("throughput_ev_per_s", "Throughput (ev/s)", "Throughput vs Threads", "throughput_vs_threads.png", max_threads=64)
make_plot("total_real", "Total Real Time per Event", "Total Real Time per Event vs Threads", "total_real_vs_threads.png", max_threads=64)
make_plot("total_cpu", "Total CPU Time per Event", "Total CPU Time per Event vs Threads", "total_cpu_vs_threads.png", max_threads=64)
make_plot("recv_real", "Total real time spent inside receiver modules", "Total time spent receiving vs Threads", "avg_receive_real.png", max_threads=64)
make_plot("send_real", "Total real time spent inside sender modules", "Total time spent sending vs Threads", "avg_send_real.png", max_threads=64)



print("✅ Plots saved to:", output_dir)
