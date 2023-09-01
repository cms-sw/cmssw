#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"

function doTest() {
	TEST="$1"
	CMD="$2"
	PATTERN="$3"
	PATTERN2="$4"
	LOG="log_test$TEST.log"
	$CMD >& $LOG || die "Test $TEST: failure running $CMD" $?
	(head -n 1 $LOG | grep -qF "$PATTERN") || die "Test $TEST: incorrect output from $CMD" $?
	if [ -n "$PATTERN2" ]; then
		(grep -qF "$PATTERN2" $LOG) || die "Test $TEST: incorrect output from $CMD" $?
	fi
}

# test w/ no args
doTest 1 "cmsRun ${LOCAL_TEST_DIR}/test_argv.py" "['${LOCAL_TEST_DIR}/test_argv.py']"

# test w/ cmsRun args
doTest 2 "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_argv.py" "['${LOCAL_TEST_DIR}/test_argv.py']" "setting # threads 2"

# test w/ python args
doTest 3 "cmsRun ${LOCAL_TEST_DIR}/test_argv.py foo" "['${LOCAL_TEST_DIR}/test_argv.py', 'foo']"

# test w/ cmsRun & python args
doTest 4 "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_argv.py foo" "['${LOCAL_TEST_DIR}/test_argv.py', 'foo']" "setting # threads 2"
