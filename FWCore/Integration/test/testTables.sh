#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/testTableTest_cfg.py || die 'Failed in testTableTest_cfg.py' $?
