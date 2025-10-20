#!/bin/bash

# Number of runs per test (first can be treated as warm-up)
runs=5

# Threads/Streams combinations to test
thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "24:24" "32:32")

# Scripts to run
script_local="hlt_local.py"
script_remote="hlt_remote.py"

# Base directory for logs
BASE_DIR="../../test_results/ompi5/milan-genoa_ucx_t-s-c"
mkdir -p "$BASE_DIR"

# Hostnames
remote_host="gputest-genoa-02"
local_host="gputest-milan-02"

# Interfaces and UCX devices
remote_iface="enp34s0f0"
local_iface="eno8303"
remote_ucx_dev="mlx5_4"
local_ucx_dev="mlx5_3"

# Resolve absolute base dir (safe for remote mkdir)
absolute_base_dir=$(realpath "$BASE_DIR")

for combo in "${thread_stream_combos[@]}"; do
    IFS=':' read -r threads streams <<< "$combo"

    end_core=$((threads - 1))

    TEST_DIR="$BASE_DIR/test_t${threads}s${streams}"
    absolute_test_dir="$absolute_base_dir/test_t${threads}s${streams}"

    mkdir -p "$TEST_DIR"

    # Create the directory remotely
    echo "Creating directory on remote: $absolute_test_dir"
    ssh "$remote_host" "mkdir -p $absolute_test_dir"

    echo "=== Running tests with ${threads} threads, ${streams} streams, CPUs: 0-$end_core ==="

    for i in $(seq 1 $runs); do
        echo "Run #$i for t${threads}s${streams} on CPU list: 0-$end_core"

        # Define environment variables for this run
        run_id=$i
        exp_threads=$threads
        exp_streams=$streams
        exp_name="milan-genoa_t${threads}s${streams}_r${i}"
        exp_output_dir="$absolute_test_dir"
        throughput_log_file="$absolute_base_dir/throughputs.txt"

        # Launch processes
        cmsenv_mpirun \
            --mca oob_tcp_if_exclude enp4s0f4u1u2c2 \
            -mca orte_base_help_aggregate 0 \
            -mca pml ucx \
            -H $remote_host \
            -np 1 \
            -x LD_LIBRARY_PATH \
            -x OMPI_MCA_ucx_net_devices=$remote_ucx_dev \
            -x RUN_ID=$run_id \
            -x EXPERIMENT_THREADS=$exp_threads \
            -x EXPERIMENT_STREAMS=$exp_streams \
            -x EXPERIMENT_NAME=$exp_name \
            -x EXPERIMENT_OUTPUT_DIR=$exp_output_dir \
            -x THROUGHPUT_LOG_FILE=$throughput_log_file \
            -bind-to none numactl \
            --physcpubind=0-"${end_core}" \
            cmsRun "$script_remote" \
            : \
            -H $local_host \
            -np 1 \
            -x OMPI_MCA_ucx_net_devices=$local_ucx_dev \
            -x RUN_ID=$run_id \
            -x EXPERIMENT_THREADS=$exp_threads \
            -x EXPERIMENT_STREAMS=$exp_streams \
            -x EXPERIMENT_NAME=$exp_name \
            -x EXPERIMENT_OUTPUT_DIR=$exp_output_dir \
            -x THROUGHPUT_LOG_FILE=$throughput_log_file \
            -bind-to none numactl \
            --physcpubind=0-"${end_core}" \
            cmsRun "$script_local"

    done

    echo "Completed tests for threads=$threads, streams=$streams"
done

echo "All cross-machine tests completed!"