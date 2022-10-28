#!/bin/bash

## source environment (get ROOT)
source xeon_scripts/init-env.sh

## Command line config
infile_name=${1:-"benchmark2_SKL-SP_results.txt"}
outfile_name=${2:-"sklsp"}
graph_label=${3:-""}

## Run little macro
./plotting/plotThroughput.py ${infile_name} ${outfile_name} ${graph_label}
