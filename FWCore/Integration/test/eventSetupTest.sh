#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupTest_cfg.py || die 'Failed in EventSetupTest_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupAppendLabelTest_cfg.py || die 'Failed in EventSetupAppendLabelTest_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupTest2_cfg.py || die 'Failed in EventSetupTest2_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupTest2_cfg.py || die 'Failed in EventSetupAppendLabelTest2_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupForceCacheClearTest_cfg.py || die 'Failed in EventSetupForceCacheClearTest_cfg.py' $?

echo testESProductHost
cmsRun --parameter-set ${LOCAL_TEST_DIR}/ESProductHostTest_cfg.py || die 'Failed in ESProductHostTest_cfg.py' $?

popd
