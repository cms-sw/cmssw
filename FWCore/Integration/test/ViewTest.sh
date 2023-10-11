#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/ViewTest_cfg.py || die 'Failed in ViewTest_cfg.py' $?
