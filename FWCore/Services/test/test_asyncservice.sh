#!/bin/bash

function die { cat log.txt; echo $1: status $2 ;  exit $2; }

CONF=${SCRAM_TEST_PATH}/test_asyncservice_cfg.py
# Normal behavior
echo "cmsRun ${CONF}"
cmsRun ${CONF} > log.txt 2>&1 || die "Failure using ${CONF}" $?

# Framework emits early termination signal, AsyncService should disallow run() calls
echo "cmsRun ${CONF} --earlyTermination"
cmsRun ${CONF} --earlyTermination > log.txt 2>&1
RET=$?
if [ "${RET}" == "0" ]; then
    cat log.txt
    die "${CONF} --earlyTermination succeeded while it was expected to fail" 1
fi
grep -q "ZombieKillerService" log.txt && die "${CONF} --earlyTermination was killed by ZombieKillerService, while the job should have failed by itself" 1
grep -q "AsyncCallNotAllowed" log.txt || die "${CONF} --earlyTermination did not fail with AsyncCallNotAllowed" $?
grep -q "Framework is shutting down, further run() calls are not allowed" log.txt || die "${CONF} --earlyTermination did not contain expected earlyTermination message" $?

# Another module throws an exception while an asynchronous function is
# running, ensure the framework to keep the data processing open until
# all asynchronous functions have returned
echo "cmsRun ${CONF} --exception"
cmsRun ${CONF} --exception > log.txt 2>&1
RET=$?
if [ "${RET}" == "0" ]; then
    cat log.txt
    die "${CONF} --exception succeeded while it was expected to fail" 1
fi
grep -q "ZombieKillerService" log.txt && die "${CONF} --exception was killed by ZombieKillerService" 1
grep -q "MoreExceptions:  AfterModEndJob" log.txt && die "${CONF} --exception threw an unexpected exception in EndJob" 1
grep -q "Intentional 'NotFound' exception for testing purposes" log.txt || die "${CONF} --exception failed in unexpected way" $?

exit 0
