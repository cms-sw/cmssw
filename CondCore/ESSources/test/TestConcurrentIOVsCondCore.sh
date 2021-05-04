#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

echo cmsRun TestConcurrentIOVsCondCore_cfg.py
cmsRun --parameter-set ${LOCAL_TEST_DIR}/TestConcurrentIOVsCondCore_cfg.py > TestConcurrentIOVsCondCoreCout.log || die 'Failed in TestConcurrentIOVsCondCore_cfg.py' $?
grep TestConcurrentIOVsCondCore TestConcurrentIOVsCondCoreCout.log > TestConcurrentIOVsCondCore.log
diff ${LOCAL_TEST_DIR}/unit_test_outputs/TestConcurrentIOVsCondCore.log TestConcurrentIOVsCondCore.log || die "comparing TestConcurrentIOVsCondCore.log" $?

popd
