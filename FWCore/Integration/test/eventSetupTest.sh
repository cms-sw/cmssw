#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }
function diecat { echo "$1: status $2, log" ;  cat $3; exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

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

echo testConcurrentIOVsESConcurrentSource_cfg.py
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testConcurrentIOVsESConcurrentSource_cfg.py || die 'Failed in testConcurrentIOVsESConcurrentSource_cfg.py' $?

echo EventSetupTestCurrentProcess_cfg.py
cmsRun ${LOCAL_TEST_DIR}/EventSetupTestCurrentProcess_cfg.py || die 'Failed in EventSetupTestCurrentProcess_cfg.py' $?

echo EventSetupIncorrectConsumes_cfg.py
cmsRun ${LOCAL_TEST_DIR}/EventSetupIncorrectConsumes_cfg.py &> testEventSetupIncorrectConsumes.txt && die 'Failed EventSetupIncorrectConsumes_cfg.py, the configuration succeeded while it should have failed' 1
grep "A module declared it consumes an EventSetup product after its constructor" testEventSetupIncorrectConsumes.txt >/dev/null || diecat 'Failed EventSetupIncorrectConsumes_cfg.py, the configuration failed but in an unexpected way' $? testEventSetupIncorrectConsumes.txt

echo testConcurrentIOVsAndRuns_cfg.py
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testConcurrentIOVsAndRuns_cfg.py || die 'Failed in testConcurrentIOVsAndRuns_cfg.py' $?

echo testConcurrentIOVsAndRunsRead_cfg.py
cmsRun --parameter-set ${LOCAL_TEST_DIR}/testConcurrentIOVsAndRunsRead_cfg.py || die 'Failed in testConcurrentIOVsAndRunsRead_cfg.py' $?
