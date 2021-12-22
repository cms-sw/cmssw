#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=src/CondTools/CTPPS/test

cmsRun $TEST_DIR/ppsTimingCalibrationLUTWriter_cfg.py || die "LUT writer failed" $?
echo "LUT writer succeeded"
cmsRun $TEST_DIR/ppsTimingCalibrationLUTAnalyzer_cfg.py || die "LUT analyzer failed" $?
echo "LUT analyzer succeeded"

