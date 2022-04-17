#! /bin/bash

useARCH=${1:-4}

# Final cleanup script for benchmarks
if [[ ${useARCH} -eq 3 ]] || [[ ${useARCH} -eq 4 ]] 
then
rm -rf benchmark_knl_dump.txt benchmark_snb_dump.txt
fi
if [[ ${useARCH} -eq 1 ]] || [[ ${useARCH} -eq 2 ]] || [[ ${useARCH} -eq 4 ]]
then
rm -rf benchmark_lnx-g_dump.txt benchmark_lnx-s_dump.txt
fi

rm -rf log_*.txt
rm -rf *.root
rm -rf *.png
rm -rf validation_*
