#!/bin/bash

# Communication mechanisms to test
comm_types=("xpmem") 

# Number of runs per test
runs=6  # 4 actual runs + 1 warmup run

# Threads/Streams combinations to test
thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "32:32" "64:64")

# Script names
local_script_name="hlt_local.py"
remote_script_name="hlt_remote.py"

# Base directory for logs
BASE_DIR="../../test_results/synchronous_thread-stream-core-all-on-1"
mkdir -p "$BASE_DIR"

# Define NUMA CPU mapping (flattened)
declare -A numa_cpu_map=(
    [0]="0 1 2 3 4 5 6 7 64 65 66 67 68 69 70 71"
    [1]="8 9 10 11 12 13 14 15 72 73 74 75 76 77 78 79"
    [2]="16 17 18 19 20 21 22 23 80 81 82 83 84 85 86 87"
    [3]="24 25 26 27 28 29 30 31 88 89 90 91 92 93 94 95"
    [4]="32 33 34 35 36 37 38 39 96 97 98 99 100 101 102 103"
    [5]="40 41 42 43 44 45 46 47 104 105 106 107 108 109 110 111"
    [6]="48 49 50 51 52 53 54 55 112 113 114 115 116 117 118 119"
    [7]="56 57 58 59 60 61 62 63 120 121 122 123 124 125 126 127"
)

# Preferred NUMA node ordering
ordered_numa_nodes=(6 7 5 4 3 2 1 0)
all_cpus=()
for node in "${ordered_numa_nodes[@]}"; do
    all_cpus+=(${numa_cpu_map[$node]})
done

# Local vs remote CPU pool: split first 64 cores to remote, next 64 to local
remote_cores=()
for i in 0 1 2 3; do
    remote_cores+=(${numa_cpu_map[$i]})
done

local_cores=()
for i in 6 7 5 4; do
    local_cores+=(${numa_cpu_map[$i]})
done

for comm in "${comm_types[@]}"; do
    for combo in "${thread_stream_combos[@]}"; do
        IFS=':' read -r threads streams <<< "$combo"

        # Determine how many cores each side needs
        local_needed=$streams
        remote_needed=$streams

        # Use overlap at 128
                # Use overlap at 128
        if [[ "$streams" -eq 128 ]]; then
            remote_bind_list=$(IFS=,; echo "${all_cpus[*]:0:128}")
            local_bind_list=$remote_bind_list
        elif [[ "$streams" -eq 96 ]]; then
            remote_bind_list=$(IFS=,; echo "${all_cpus[*]:0:96}")
            local_bind_list=$(IFS=,; echo "${all_cpus[*]:32:96}")
        else
            remote_bind_list=$(IFS=,; echo "${remote_cores[*]:0:$remote_needed}")
            local_bind_list=$(IFS=,; echo "${local_cores[*]:0:$local_needed}")
        fi

        TEST_DIR="$BASE_DIR/test_${comm}_t${threads}s${streams}_pinned"
        mkdir -p "$TEST_DIR"

        echo "=== Running tests for $comm with ${threads} threads, ${streams} streams ==="

        for i in $(seq 1 $runs); do
            echo "Run #$i for $comm / t${threads}s${streams}"

            export RUN_ID=$i
            export COMM_LAYER_NAME=$comm
            export EXPERIMENT_THREADS=$threads
            export EXPERIMENT_STREAMS=$streams
            export EXPERIMENT_NAME="comm_${comm}_t${threads}s${streams}_r${i}"
            export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"
            export THROUGHPUT_LOG_FILE="$BASE_DIR/throughputs.txt"
            export OMPI_MCA_hwloc_base_binding_policy=none

            # Set OpenMPI parameters
            case "$comm" in
                "xpmem")
                    export OMPI_MCA_pml=ob1
                    export OMPI_MCA_btl_vader_single_copy_mechanism=xpmem
                    export OMPI_MCA_btl=self,vader
                    ;;
                "cma")
                    export OMPI_MCA_pml=ob1
                    export OMPI_MCA_btl_vader_single_copy_mechanism=cma
                    export OMPI_MCA_btl=self,vader
                    ;;
                "tcp")
                    export OMPI_MCA_pml=ob1
                    export OMPI_MCA_btl=self,tcp
                    ;;
                "ucx")
                    export OMPI_MCA_pml=ucx
                    export OMPI_MCA_btl=self
                    ;;
                *)
                    echo "Unknown communication mechanism: $comm"
                    exit 1
                    ;;
            esac

            echo "Verifying MPI config for $comm"
            mpirun --version
            ompi_info --all | grep -E "btl|pml"

            # Print resolved command
            echo "mpirun -np 1 numactl --physcpubind=$remote_bind_list cmsRun $remote_script_name : -np 1 numactl --physcpubind=$local_bind_list cmsRun $local_script_name"


            # Run with core pinning for each process
            mpirun -np 1 numactl --physcpubind="$local_bind_list" cmsRun "$remote_script_name" \
                : -np 1 numactl --physcpubind="$local_bind_list" cmsRun "$local_script_name"
        done

        echo "Completed tests for $comm with $threads threads, $streams streams"
    done
done

echo "All pinned dual-process tests completed!"
