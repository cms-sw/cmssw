#!/bin/bash

# Communication mechanisms to test
comm_types=("xpmem") 

# Number of runs per test
runs=6  # 4 actual runs + 1 warmup run

# Threads/Streams combinations to test
# thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "32:32" "64:64")
thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "24:24" "32:32" "48:48" "64:64")

# Script names
local_script_name="hlt_local.py"
remote_script_name="hlt_remote.py"

# Base directory for logs
BASE_DIR="../../test_results/synchronous_thread-stream-core_1_socket_oversub"
mkdir -p "$BASE_DIR"

# Define NUMA CPU mapping (flattened)
# Define NUMA CPU mapping
declare -A numa_cpu_map=(
    [6]="48 49 50 51 52 53 54 55 112 113 114 115 116 117 118 119"
    [7]="56 57 58 59 60 61 62 63 120 121 122 123 124 125 126 127"
    [5]="40 41 42 43 44 45 46 47 104 105 106 107 108 109 110 111"
    [4]="32 33 34 35 36 37 38 39 96 97 98 99 100 101 102 103"
    [3]="24 25 26 27 28 29 30 31 88 89 90 91 92 93 94 95"
    [2]="16 17 18 19 20 21 22 23 80 81 82 83 84 85 86 87"
    [1]="8 9 10 11 12 13 14 15 72 73 74 75 76 77 78 79"
    [0]="0 1 2 3 4 5 6 7 64 65 66 67 68 69 70 71"
)

# Only use nodes 6,7,5,4
ordered_numa_nodes=(6 7 5 4)

# Rebuild all_cpus with real cores first, then hyperthreads
real_cores=()
ht_cores=()
for node in "${ordered_numa_nodes[@]}"; do
    cpu_list=(${numa_cpu_map[$node]})
    real_cores+=( "${cpu_list[@]:0:8}" )
    ht_cores+=( "${cpu_list[@]:8:8}" )
done
all_cpus=( "${real_cores[@]}" "${ht_cores[@]}" )


for comm in "${comm_types[@]}"; do
    for combo in "${thread_stream_combos[@]}"; do
        IFS=':' read -r threads streams <<< "$combo"

        total_cores=$streams  # only streams determine how many physical cores we bind to

        # Pick just enough CPUs to cover total_cores
        selected_cpus=("${all_cpus[@]:0:$total_cores}")
        cpu_list=$(IFS=,; echo "${selected_cpus[*]}")


        TEST_DIR="$BASE_DIR/test_${comm}_t${threads}s${streams}_pinned"
        mkdir -p "$TEST_DIR"

        echo "=== Running tests for $comm with ${threads} threads, ${streams} streams, cpus: $cpu_list ==="

        for i in $(seq 1 $runs); do
            echo "Run #$i for $comm / t${threads}s${streams}"

            export RUN_ID=$i
            export COMM_LAYER_NAME=$comm
            export EXPERIMENT_THREADS=$threads
            export EXPERIMENT_STREAMS=$streams
            export EXPERIMENT_NAME="comm_${comm}_t${threads}s${streams}_r${i}"
            export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"
            export THROUGHPUT_LOG_FILE="$BASE_DIR/throughputs.txt"
            # export OMPI_MCA_hwloc_base_binding_policy=none

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
            echo "cpu_list is: '$cpu_list'"

            # Run with core pinning for each process
            mpirun -np 1 -bind-to none numactl --physcpubind="${cpu_list}" cmsRun "$remote_script_name" \
                : -np 1 -bind-to none numactl --physcpubind="${cpu_list}" cmsRun "$local_script_name"
        done

        echo "Completed tests for $comm with $threads threads, $streams streams"
    done
done

echo "All pinned dual-process tests completed!"
