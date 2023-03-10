#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${SCRAM_TEST_PATH}/parentlessTest_cfg.py || die 'Failed in parentlessTest_cfg.py' $?
