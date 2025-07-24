#!/bin/bash

function die { cat $1; echo $2: status $3 ;  exit 32; }

CONF=${SCRAM_TEST_PATH}/test_asyncservice_cfg.py
# Normal behavior
echo "cmsRun ${CONF}"
cmsRun ${CONF} > log_normal.txt 2>&1 || die log_normal.txt "Failure using ${CONF}" $?

# Framework emits early termination signal, AsyncService should disallow run() calls
echo "cmsRun ${CONF} --earlyTermination"
cmsRun ${CONF} --earlyTermination > log_earlyTermination.txt 2>&1 && die log_earlytermination.txt "${CONF} --earlyTermination succeeded while it was expected to fail" 1
grep -q "ZombieKillerService" log_earlyTermination.txt && die log_earlyTermination.txt "${CONF} --earlyTermination was killed by ZombieKillerService, while the job should have failed by itself" 1
grep -q "AsyncCallNotAllowed" log_earlyTermination.txt || die log_earlyTermination.txt "${CONF} --earlyTermination did not fail with AsyncCallNotAllowed" $?
grep -q "Framework is shutting down, further run() calls are not allowed" log_earlyTermination.txt || die log_earlyTermination.txt "${CONF} --earlyTermination did not contain expected earlyTermination message" $?

# Another module throws an exception while an asynchronous function is
# running, ensure the framework to keep the data processing open until
# all asynchronous functions have returned
echo "cmsRun ${CONF} --exception"
cmsRun ${CONF} --exception > log_exception.txt 2>&1 && die log_exception.txt "${CONF} --exception succeeded while it was expected to fail" 1
grep -q "ZombieKillerService" log_exception.txt && die log_exception.txt "${CONF} --exception was killed by ZombieKillerService" 1
grep -q "MoreExceptions:  AfterModEndJob" log_exception.txt && die log_exception.txt "${CONF} --exception threw an unexpected exception in EndJob" 1
grep -q "Intentional 'NotFound' exception for testing purposes" log_exception.txt || die log_exception.txt "${CONF} --exception failed in unexpected way" $?

exit 0
