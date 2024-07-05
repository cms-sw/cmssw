#!/bin/bash

# Check if the key argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <key>"
    exit 1
fi

# Extract the key from the command line argument
key="$1"

# Define a function to run the python command
run_python_command() {
    function die { echo $1: status $2 ; exit $2; }

    entry="$1"
    key="$2"

    # Check conditions to skip certain combinations
    if [[ "$entry" == *visualization-live_cfg.py* && ( "$key" == "pp_run_stage1" || "$key" == "cosmic_run_stage1" || "$key" == "hpu_run" ) ]]; then
        echo "===== Skipping Test \"python3 $entry runkey=$key\" ===="
        return
    fi

    # Otherwise, proceed with the test
    echo "===== Test \"python3 $entry runkey=$key\" ===="
    (python3 "$entry" runkey="$key" > /dev/null) 2>&1 || die "Failure using python3 $entry" $?
}

# Run the tests for the specified key
echo "Running tests for key: $key"
for entry in "${CMSSW_BASE}/src/DQM/Integration/python/clients/"*"-live_cfg.py"; do
    run_python_command "$entry" "$key"
done

# All tests passed
echo "All tests passed!"
