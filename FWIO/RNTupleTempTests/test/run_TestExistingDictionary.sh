#!/bin/bash

test=testExistingDictionaryChecking

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  echo "*************************************************"
  echo "Producer"
  cmsRun ${LOCAL_TEST_DIR}/${test}_cfg.py || die "cmsRun ${test}_cfg.py 1" $?

  echo "*************************************************"
  echo "Consumer"
  cmsRun ${LOCAL_TEST_DIR}/${test}Read_cfg.py || die "cmsRun ${test}Read_cfg.py 1" $?

