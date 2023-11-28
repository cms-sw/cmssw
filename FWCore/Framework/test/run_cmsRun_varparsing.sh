#!/bin/bash

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
source "${LOCAL_TEST_DIR}/help_cmsRun_tests.sh"

# test cmsRun help
doTest 1 "cmsRun --help ${LOCAL_TEST_DIR}/test_varparsing.py" "cmsRun [options] [--] config_file [python options]"

# test python help
doTest 2 "cmsRun ${LOCAL_TEST_DIR}/test_varparsing.py --help" "Singletons:
  maxEvents: 1
           - max events to process
  threads  : 1
           - number of threads
Lists:

Options:
        help           : This screen
        multipleAssign : Allows singletons to have multiple assignments
        print          : Prints out current values
        XXX_clear      : Clears list named 'XXX'
"

# test nonexistent flag
TEST=3
CMD="cmsRun ${LOCAL_TEST_DIR}/test_varparsing.py nonexistent=foo"
$CMD >& log_test$TEST.log && die "Test $TEST: no error from $CMD" 1
(head -n 1 log_test$TEST.log | grep -qF "Error:  'nonexistent' not registered.") || die "Test $TEST: incorrect output from $CMD" $?

# test cmsRun args
TEST4_OUT1="Singletons:
  maxEvents: 1
           - max events to process
  threads  : 1
           - number of threads
Lists:
"
TEST4_OUT2="setting # threads 2"
doTest 4 "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_varparsing.py" "$TEST4_OUT1" "$TEST4_OUT2"

# test python args
TEST=5
TEST5_OUT1="Singletons:
  maxEvents: 1
           - max events to process
  threads  : 2
           - number of threads
Lists:
"
doTest $TEST "cmsRun ${LOCAL_TEST_DIR}/test_varparsing.py threads=2" "$TEST5_OUT1"
(grep -vqF "$TEST4_OUT2" log_test$TEST.log) || die "Test $TEST: incorrect output from $CMD" $?

# test cmsRun args and python args together
TEST=6
TEST6_OUT1="Singletons:
  maxEvents: 1
           - max events to process
  threads  : 3
           - number of threads
Lists:
"
TEST6_OUT2="setting # threads 2"
doTest $TEST "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_varparsing.py threads=3" "$TEST6_OUT1" "$TEST6_OUT2"
