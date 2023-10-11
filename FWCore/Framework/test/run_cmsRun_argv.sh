#!/bin/bash

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
source "${LOCAL_TEST_DIR}/help_cmsRun_tests.sh"

# test w/ no args
doTest 1 "cmsRun ${LOCAL_TEST_DIR}/test_argv.py" "['${LOCAL_TEST_DIR}/test_argv.py']"

# test w/ cmsRun args
doTest 2 "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_argv.py" "['${LOCAL_TEST_DIR}/test_argv.py']" "setting # threads 2"

# test w/ python args
doTest 3 "cmsRun ${LOCAL_TEST_DIR}/test_argv.py foo" "['${LOCAL_TEST_DIR}/test_argv.py', 'foo']"

# test w/ cmsRun & python args
doTest 4 "cmsRun -n 2 ${LOCAL_TEST_DIR}/test_argv.py foo" "['${LOCAL_TEST_DIR}/test_argv.py', 'foo']" "setting # threads 2"
