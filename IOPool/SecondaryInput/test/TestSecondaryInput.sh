#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PreSecondaryInputTest2_cfg.py || die 'Failure using PreSecondaryInputTest2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PreSecondaryInputTest_cfg.py || die 'Failure using PreSecondaryInputTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/SecondaryInputTest_cfg.py || die 'Failure using SecondaryInputTest_cfg.py' $?
