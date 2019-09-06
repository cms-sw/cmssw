#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupTest_cfg.py || die 'Failed in EventSetupTest_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupAppendLabelTest_cfg.py || die 'Failed in EventSetupAppendLabelTest_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupTest2_cfg.py || die 'Failed in EventSetupTest2_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupForceCacheClearTest_cfg.py || die 'Failed in EventSetupForceCacheClearTest_cfg.py' $?

echo testESProductHost
cmsRun --parameter-set ${LOCAL_TEST_DIR}/ESProductHostTest_cfg.py || die 'Failed in ESProductHostTest_cfg.py' $?

echo testConcurrentIOVs
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testConcurrentIOVs_cfg.py || die 'Failed in testConcurrentIOVs_cfg.py' $?

echo testConcurrentIOVsLegacy
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testConcurrentIOVsLegacy_cfg.py || die 'Failed in testConcurrentIOVsLegacy_cfg.py' $?

echo testAllowConcurrentIOVs_cfg
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testAllowConcurrentIOVs_cfg.py || die 'Failed in testAllowConcurrentIOVs_cfg.py' $?

echo testConcurrentIOVsForce_cfg
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testConcurrentIOVsForce_cfg.py || die 'Failed in testConcurrentIOVsForce_cfg.py' $?

echo testEventSetupRunLumi_cfg
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testEventSetupRunLumi_cfg.py || die 'Failed in testEventSetupRunLumi_cfg.py' $?

echo testConcurrentIOVsESSource_cfg.py
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testConcurrentIOVsESSource_cfg.py || die 'Failed in testConcurrentIOVsESSource_cfg.py' $?

popd
