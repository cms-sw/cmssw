#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/inputSourceTest_cfg.py || die 'Failed in inputSourceTest_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/testLateLumiClosure_cfg.py || die 'Failed in testLateLumiClosure_cfg.py' $?
