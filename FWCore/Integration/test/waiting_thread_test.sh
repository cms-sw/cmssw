#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

echo "cmsRun acquireTest_cfg.py"
cmsRun --parameter-set ${LOCAL_TEST_DIR}/acquireTest_cfg.py || die 'Failed in acquireTest_cfg.py' $?
