#!/bin/bash

test=testtestDoc
function die { echo Failure $1: status $2 ; exit $2 ; }
pushd ${LOCAL_TMP_DIR}
  export PATH=${LOCAL_TOP_DIR}/test/${SCRAM_ARCH}/:${PATH}
  cd ${LOCAL_TEST_DIR}
  echo ${test} testDoc ------------------------------------------------------------
  testDocProg=testDoc
  for scriptDir in $CMSSW_BASE $CMSSW_RELEASE_BASE $CMSSW_FULL_RELEASE_BASE ; do
    if [ -x $scriptDir/test/${SCRAM_ARCH}/testDoc ] ; then 
      testDocProg=$scriptDir/test/${SCRAM_ARCH}/testDoc
      echo Found $testDocProg
      break
    fi
  done
  $testDocProg testConfiguration.xml || die "testDoc" $?
  popd
exit 0
