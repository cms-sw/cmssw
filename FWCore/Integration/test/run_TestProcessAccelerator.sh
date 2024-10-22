#!/bin/bash

test=testProcessAccelerator
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

function die { echo Failure $1: status $2 ; exit $2 ; }

echo "*************************************************"
echo "accelerators=*"
cmsRun ${LOCAL_TEST_DIR}/${test}_cfg.py || die "cmsRun ${test}_cfg.py" $?

echo "*************************************************"
echo "accelerators=*, enableTest2"
cmsRun ${LOCAL_TEST_DIR}/${test}_cfg.py --enableTest2 || die "cmsRun ${test}_cfg.py --enableTest2" $?

echo "*************************************************"
echo "accelerators=test1"
cmsRun ${LOCAL_TEST_DIR}/${test}_cfg.py --accelerators=test1 || die "cmsRun ${test}_cfg.py --accelerators=test1" $?

echo "*************************************************"
echo "accelerators=test2"
cmsRun -j testProcessAccelerators_jobreport.xml ${LOCAL_TEST_DIR}/${test}_cfg.py --accelerators=test2 && die "cmsRun ${test}_cfg.py --accelerators=test2 did not fail" 1
EXIT_CODE=$(edmFjrDump --exitCode testProcessAccelerators_jobreport.xml)
if [ "x${EXIT_CODE}" != "x8035" ]; then
    echo "ProcessAccelerator test for unavailable accelerator reported exit code ${EXIT_CODE} which is different from the expected 8035"
    exit 1
fi

echo "*************************************************"
echo "accelerators=test1, enableTest2"
cmsRun ${LOCAL_TEST_DIR}/${test}_cfg.py --accelerators=test1 --enableTest2 || die "cmsRun ${test}_cfg.py --accelerators=test1 --enableTest2" $?

echo "*************************************************"
echo "accelerators=test2, enableTest2"
cmsRun ${LOCAL_TEST_DIR}/${test}_cfg.py --accelerators=test2 --enableTest2 || die "cmsRun ${test}_cfg.py --accelerators=test2 --enableTest2" $?
