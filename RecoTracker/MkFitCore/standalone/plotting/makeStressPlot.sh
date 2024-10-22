#!/bin/bash

## source environment (get ROOT)
source xeon_scripts/init-env.sh

## Command line config
infile_name=${1:-"stress_test_SKL-SP_results.txt"}
graph_label=${2:-"[Turbo=OFF(Long)]"}
outfile_name=${3:-"noturbo1_long.pdf"}

## reduce stress test results to results used in macro only
tmp_infile_name="tmp_results.txt"
> "${tmp_infile_name}"

grep "SSE3" "${infile_name}" >> "${tmp_infile_name}"
grep "AVX2" "${infile_name}" >> "${tmp_infile_name}"
grep "AVX512" "${infile_name}" >> "${tmp_infile_name}"

## Run little macro
root -l -b -q plotting/plotStress.C\(\"${tmp_infile_name}\",\"${graph_label}\",\"${outfile_name}\"\)

## remove tmp file
rm "${tmp_infile_name}"
