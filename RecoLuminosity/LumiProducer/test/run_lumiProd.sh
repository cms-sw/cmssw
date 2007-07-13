#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/testLumiProd.cfg || die "cmsRun testLumiProd.cfg" $?

popd

exit 0
