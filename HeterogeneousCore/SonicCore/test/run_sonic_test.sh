#!/bin/bash

function die { echo "$1": status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

TESTCFG=sonicTest_cfg.py
echo "cmsRun $TESTCFG"
cmsRun --parameter-set ${LOCAL_TEST_DIR}/${TESTCFG} || die "Failed in $TESTCFG" $?

popd
