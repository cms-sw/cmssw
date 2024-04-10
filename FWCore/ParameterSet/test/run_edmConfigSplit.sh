#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/ParameterSet/test"

function doTest() {
	TEST="$1"
	CMD="$2"
	REFOUT1="${LOCAL_TEST_DIR}/unit_test_outputs/split.py"
	REFOUT2="${LOCAL_TEST_DIR}/unit_test_outputs/m1a_cfi.py"
	LOG="log_test$TEST.log"
	$CMD >& $LOG || die "Test $TEST: failure running $CMD" $?
	(diff $REFOUT1 $LOG) || die "Test $TEST: incorrect output from $CMD" $?
	(diff $REFOUT2 m1a_cfi.py) || die "Test $TEST: incorrect output from $CMD" $?
}

# test edmConfigSplit w/ argparse
doTest 1 "edmConfigSplit ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2"

# test edmConfigSplit w/ varparsing
OUT=dump_varparsing.py
doTest 2 "edmConfigSplit ${LOCAL_TEST_DIR}/test_varparsing.py output=foo intprod=2"
