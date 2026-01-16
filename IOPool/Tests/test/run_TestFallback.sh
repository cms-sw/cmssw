#!/bin/bash

test=testNoProcessFallback

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  echo "testFallback1"
  cmsRun ${LOCAL_TEST_DIR}/${test}1_cfg.py > ${test}1.log 2>/dev/null || die "cmsRun ${test}1_cfg.py" $?

  echo "testFallback2"
  cmsRun ${LOCAL_TEST_DIR}/${test}2_cfg.py > ${test}2.log 2>/dev/null || die "cmsRun ${test}2_cfg.py" $?

  echo "testFallback3"
  cmsRun ${LOCAL_TEST_DIR}/${test}3_cfg.py || die "cmsRun ${test}3_cfg.py" $?

  echo "testFallbackNoCurrent"
  cmsRun ${LOCAL_TEST_DIR}/${test}NoCurrent_cfg.py || die "cmsRun ${test}NoCurrent_cfg.py" $?

exit 0
