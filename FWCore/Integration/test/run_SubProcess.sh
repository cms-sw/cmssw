#!/bin/bash

test=testSubProcess

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/${test}_cfg.py || die "cmsRun ${test}_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/readSubProcessOutput_cfg.py || die "cmsRun readSubProcessOutput_cfg.py" $?

popd

exit 0
