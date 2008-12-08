#!/bin/bash

test=testParameterSet

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo ${test} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}_cfg.py 2> ${test}.txt
  grep "Illegal parameter" ${test}.txt || die "cmsRun ${test}_cfg.py" $?

popd

exit 0
