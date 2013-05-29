#!/bin/bash

test=testGetBy

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/${test}1_cfg.py || die "cmsRun ${test}1_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}2_cfg.py || die "cmsRun ${test}2_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}3_cfg.py || die "cmsRun ${test}3_cfg.py" $?

popd

exit 0
