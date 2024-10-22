#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/PreSecondaryInputTest2_cfg.py || die 'Failure using PreSecondaryInputTest2_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PreSecondaryInputTest_cfg.py || die 'Failure using PreSecondaryInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/SecondaryInputTest_cfg.py || die 'Failure using SecondaryInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/SecondarySeqInputTest_cfg.py || die 'Failure using SecondarySeqInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/SecondaryInLumiInputTest_cfg.py || die 'Failure using SecondaryInLumiInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/SecondarySeqInLumiInputTest_cfg.py || die 'Failure using SecondarySeqInLumiInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/SecondarySpecInputTest_cfg.py || die 'Failure using SecondarySpecInputTest_cfg.py' $?

