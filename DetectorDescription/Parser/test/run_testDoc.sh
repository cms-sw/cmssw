#!/bin/bash
test=testtestDoc
function die { echo Failure $1: status $2 ; exit $2 ; }
pushd ${LOCAL_TMP_DIR}
  export mecpath=${PATH}
  test -z "${CMSSW_RELEASE_BASE}" && echo CMSSW_RELEASE_BASE is not set.
  export PATH=${CMSSW_RELEASE_BASE}/test/${SCRAM_ARCH}/:${LOCAL_TOP_DIR}/test/${SCRAM_ARCH}/:${PATH}
  cd ${LOCAL_TEST_DIR}
  echo ${test} testDoc ------------------------------------------------------------
  testDoc testConfiguration.xml || die "testDoc" $?
  export PATH=${mecpath}
  popd
exit 0
