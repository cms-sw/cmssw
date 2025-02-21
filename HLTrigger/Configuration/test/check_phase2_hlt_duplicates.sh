#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  if [ $# -gt 2 ]; then
    printf "%s\n" "=== Log File =========="
    cat $3
    printf "%s\n" "=== End of Log File ==="
  fi
  exit $2
}

# Directory for the generated files
PYTHON_FILE="Phase2_dump_cfg.py"
DUMP_FILE="Phase2_dump.py"
GROUPS_FILE="hltFindDuplicates_output/groups.txt"

# Step 1: Generate the Python file
cat << EOF > "$PYTHON_FILE"
import FWCore.ParameterSet.Config as cms
process = cms.Process("HLT")
process.load("HLTrigger.Configuration.HLT_75e33_cff")
EOF

# Step 2: Run edmConfigDump on the generated Python file
echo "Running edmConfigDump..."
edmConfigDump "$PYTHON_FILE" > "$DUMP_FILE"
if [[ $? -ne 0 ]]; then
  echo "Error: edmConfigDump failed."
  exit 1
fi

# Step 3: Check the dumped configuration for syntax errors
echo "Running compilation check..."
python3 -m py_compile "$DUMP_FILE" 2> "syntax_errors.txt"
if [[ $? -ne 0 ]]; then
  echo "Error: The dumped configuration has syntax errors."
  cat "syntax_errors.txt"
  exit 1
fi

# Step 3: Run hltFindDuplicates on the dumped configuration
echo "Running hltFindDuplicates..."
hltFindDuplicates $DUMP_FILE -v 2 \
  -o hltFindDuplicates_output &> test_hltFindDuplicates_log \
  || die 'Failure running hltFindDuplicates' $? test_hltFindDuplicates_log
if [[ $? -ne 0 ]]; then
  echo "Error: hltFindDuplicates failed."
  exit 1
fi

# Step 4: Check if groups.txt is empty
echo "Checking group file..."
if [[ ! -f "$GROUPS_FILE" ]]; then
  echo "Error: $GROUPS_FILE not found."
  exit 1
fi

if [[ -s "$GROUPS_FILE" ]]; then
  echo "Duplicates found. Contents of $GROUPS_FILE:"
  cat "$GROUPS_FILE"
  exit 1
else
  echo "No duplicates found. Exiting successfully."
  exit 0
fi
