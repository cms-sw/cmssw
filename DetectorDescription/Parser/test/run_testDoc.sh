#!/bin/bash

test=testtestDoc
function die { echo Failure $1: status $2 ; exit $2 ; }
  export PATH=${LOCAL_TOP_DIR}/test/${SCRAM_ARCH}/:${PATH}
  cd ${SCRAM_TEST_PATH}
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
