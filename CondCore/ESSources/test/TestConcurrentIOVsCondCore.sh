#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

echo cmsRun TestConcurrentIOVsCondCore_cfg.py
cmsRun --parameter-set ${SCRAM_TEST_PATH}/TestConcurrentIOVsCondCore_cfg.py || die 'Failed in TestConcurrentIOVsCondCore_cfg.py' $?
grep "TestConcurrentIOVsCondCore: " TestConcurrentIOVsCondCoreCout.log > TestConcurrentIOVsCondCore.log
diff ${SCRAM_TEST_PATH}/unit_test_outputs/TestConcurrentIOVsCondCore.log TestConcurrentIOVsCondCore.log || die "comparing TestConcurrentIOVsCondCore.log" $?
