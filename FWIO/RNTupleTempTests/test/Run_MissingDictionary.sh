#!/bin/sh
set -x
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# It is intentional that this test throws an exception. The test fails if it does not.
cmsRun ${LOCAL_TEST_DIR}/testMissingDictionaryChecking_cfg.py &> testMissingDictionaryChecking.log && die 'Failed to get exception running testMissingDictionaryChecking_cfg.py' 1
grep -q MissingDictionaryTestF testMissingDictionaryChecking.log || die 'Failed to print out exception message with missing dictionary listed' $?

#grep -w ESProducer CatcheStdException.log
