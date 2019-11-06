#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

echo cmsRun TestConcurrentIOVsCondCore_cfg.py
cmsRun --parameter-set ${LOCAL_TEST_DIR}/TestConcurrentIOVsCondCore_cfg.py > TestConcurrentIOVsCondCore.log || die 'Failed in TestConcurrentIOVsCondCore_cfg.py' $?
diff ${LOCAL_TEST_DIR}/unit_test_outputs/TestConcurrentIOVsCondCore.log TestConcurrentIOVsCondCore.log || die "comparing TestConcurrentIOVsCondCore.log" $?

popd
