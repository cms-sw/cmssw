#!/bin/bash

# Communication mechanisms to test
# comm_types=("xpmem" "cma" "tcp" "ucx")
comm_types=("xpmem" "ucx")

# Number of runs per test
runs=5  # 4 actual runs + 1 warmup run

# Threads/Streams combinations to test (e.g. "4:4" means 4 threads, 4 streams)
thread_stream_combos=("4:4" "8:8" "16:16" "32:32" "64:64" "128:128" "256:256")

# NUMA configurations: format "local_node:remote_node"
numa_pairs=("6:7")

# Corrected variable assignments (no spaces around =)
local_script_name="hlt_local.py"
remote_script_name="hlt_remote.py"

# Base directory for logs
BASE_DIR="../../test_results/synchronous_thread_dependance"
mkdir -p "$BASE_DIR"

for comm in "${comm_types[@]}"; do
    for combo in "${thread_stream_combos[@]}"; do
        IFS=':' read -r threads streams <<< "$combo"

        for numa in "${numa_pairs[@]}"; do
            IFS=':' read -r local_numa remote_numa <<< "$numa"

            TEST_DIR="$BASE_DIR/test_${comm}_t${threads}s${streams}_n${local_numa}-${remote_numa}"
            mkdir -p "$TEST_DIR"

            echo "=== Running tests for $comm with ${threads} threads, ${streams} streams, local NUMA $local_numa, remote NUMA $remote_numa ==="

            for i in $(seq 1 $runs); do
                echo "Run #$i for $comm / t${threads}s${streams} / numa $local_numa:$remote_numa"

                export RUN_ID=$i
                export COMM_LAYER_NAME=$comm
                export EXPERIMENT_THREADS=$threads
                export EXPERIMENT_STREAMS=$streams
                export EXPERIMENT_NAME="comm_${comm}_t${threads}s${streams}_n${local_numa}-${remote_numa}_r${i}"
                export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"

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

                # Launch both remote and local sides
                mpirun -np 1 numactl -N "$remote_numa" cmsRun "$remote_script_name" \
                    : -np 1 numactl -N "$local_numa" cmsRun "$local_script_name"
            done

            echo "Completed tests for $comm + threads=$threads streams=$streams numa=$local_numa:$remote_numa"
        done
    done
done

echo "All tests completed!"
