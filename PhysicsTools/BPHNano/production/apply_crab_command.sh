#!/bin/bash
# Get the command from the first argument
COMMAND=$1
# Find directories exactly two levels deep
find . -mindepth 2 -maxdepth 2 -type d | while read -r dir; do
    echo "Processing directory: $dir"
    crab $COMMAND -d "$dir"
done
