#!/bin/bash
# Check Pool output layout and fast cloning against generated inputs,
# including merge settings and mismatched branch layouts.

set -eu

function die { echo "$1: status $2"; exit "$2"; }

for MODE in input input2; do
  ARGS=()
  if [ "$MODE" = input2 ]; then ARGS=(--second); fi
  cmsRun "${SCRAM_TEST_PATH}/poolLayout_cfg.py" --mode input "${ARGS[@]}" || die "$MODE" $?
done

python3 "${SCRAM_TEST_PATH}/testPoolLayoutCheck.py" prepare || die 'prepare test input files' $?
python3 "${SCRAM_TEST_PATH}/testPoolLayoutCheck.py" record || die 'record test input properties' $?

for MODE in clone copy merge override overridefast; do
  cmsRun -j "layout_${MODE}.xml" "${SCRAM_TEST_PATH}/poolLayout_cfg.py" --mode "$MODE" || die "$MODE" $?
done

cmsRun "${SCRAM_TEST_PATH}/poolLayout_cfg.py" --mode mismatch > layout_mismatch.log 2>&1 && { cat layout_mismatch.log; die 'mismatched input unexpectedly succeeded' 1; }
grep -q 'MismatchedInputFiles' layout_mismatch.log || { cat layout_mismatch.log; die 'missing MismatchedInputFiles' 1; }
grep -q 'different ROOT split levels or basket sizes' layout_mismatch.log || { cat layout_mismatch.log; die 'wrong mismatch diagnostic' 1; }

python3 "${SCRAM_TEST_PATH}/testPoolLayoutCheck.py" check || die 'layout verification' $?
