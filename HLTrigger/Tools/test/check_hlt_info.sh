#!/bin/bash

FILE="/store/data/Run2024I/EphemeralHLTPhysics0/RAW/v1/000/386/593/00000/91a08676-199e-404c-9957-f72772ef1354.root"

EXPECTED_OUTPUT=$(cat <<EOF
process LHC (release CMSSW_14_0_15_patch1)

process HLT (release CMSSW_14_0_15_patch1)
   HLT menu:   '/cdaq/physics/Run2024/2e34/v1.4.9/HLT/V1'
   global tag: '140X_dataRun3_HLT_v3'
EOF
)

# Run hltInfo and capture its output
ACTUAL_OUTPUT=$(hltInfo "$FILE")

# Compare using diff
if diff <(echo "$ACTUAL_OUTPUT") <(echo "$EXPECTED_OUTPUT") > /dev/null; then
  echo "Output matches expected format."
  exit 0
else
  echo "Output does NOT match expected format."
  echo
  echo "---- Expected Output ----"
  echo "$EXPECTED_OUTPUT"
  echo "-----------------------"
  echo
  echo "---- Actual Output ----"
  echo "$ACTUAL_OUTPUT"
  echo "-----------------------"
  exit 1
fi
