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

# test cmsRun help
doTest 1 "cmsRun --help ${LOCAL_TEST_DIR}/test_argparse.py" "cmsRun [options] config_file [python options]"

# test python help
doTest 2 "cmsRun ${LOCAL_TEST_DIR}/test_argparse.py --help" "usage: test_argparse.py"

# test nonexistent flag
TEST=3
CMD="cmsRun ${LOCAL_TEST_DIR}/test_argparse.py --nonexistent"
$CMD >& log_test$TEST.log && die "Test $TEST: no error from $CMD" $?
(head -n 1 log_test$TEST.log | grep -qF "usage: test_argparse.py") || die "Test $TEST: incorrect output from $CMD" $?
(grep -qF -- "unrecognized arguments: --nonexistent" log_test$TEST.log) || die "Test $TEST: incorrect output from $CMD" $?

# test cmsRun args
TEST4_OUT1="Namespace(maxEvents=1, jobreport='UNSET', enablejobreport='UNSET', mode='UNSET', numThreads='UNSET', sizeOfStackForThreadsInKB='UNSET', strict='UNSET', command='UNSET')"
TEST4_OUT2="setting # threads 2"
doTest 4 "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_argparse.py" "$TEST4_OUT1" "$TEST4_OUT2"

# test python args
TEST=5
TEST5_OUT1="Namespace(maxEvents=1, jobreport='UNSET', enablejobreport='UNSET', mode='UNSET', numThreads='2', sizeOfStackForThreadsInKB='UNSET', strict='UNSET', command='UNSET')"
doTest $TEST "cmsRun ${LOCAL_TEST_DIR}/test_argparse.py -n 2" "$TEST5_OUT1"
(grep -vqF "$TEST4_OUT2" log_test$TEST.log) || die "Test $TEST: incorrect output from $CMD" $?

# test cmsRun args and python args together
TEST=6
TEST6_OUT1="Namespace(maxEvents=1, jobreport='UNSET', enablejobreport='UNSET', mode='UNSET', numThreads='3', sizeOfStackForThreadsInKB='UNSET', strict='UNSET', command='UNSET')"
TEST6_OUT2="setting # threads 2"
doTest $TEST "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_argparse.py -n 3" "$TEST6_OUT1" "$TEST6_OUT2"
