#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/ParameterSet/test"

function doTest() {
	TEST="$1"
	CMD="$2"
	OUT="$3"
	REFOUT="${LOCAL_TEST_DIR}/unit_test_outputs/dump.py"
	LOG="log_test$TEST.log"
	$CMD >& $LOG || die "Test $TEST: failure running $CMD" $?
	(diff $REFOUT $OUT) || die "Test $TEST: incorrect output from $CMD" $?
}

# test edmConfigDump w/ argparse
OUT=dump_argparse.py
doTest 1 "edmConfigDump -o $OUT ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2" $OUT

# test edmConfigDump w/ varparsing
OUT=dump_varparsing.py
doTest 2 "edmConfigDump -o $OUT ${LOCAL_TEST_DIR}/test_varparsing.py output=foo intprod=2" $OUT
