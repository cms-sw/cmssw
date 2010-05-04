#!/bin/bash
test=testtestDoc
function die { echo Failure $1: status $2 ; exit $2 ; }
pushd ${LOCAL_TMP_DIR}
   echo ${test}testD ------------------------------------------------------------
   cd ${LOCAL_TEST_DIR}
   testDoc || die "cmsRun ${test}PROD1_cfg.py" $?
popd
exit 0
