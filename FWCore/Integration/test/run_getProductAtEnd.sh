#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  cmsRun ${LOCAL_TEST_DIR}/testGetProductAtEnd_cfg.py || die "cmsRun testGetProductAtEnd_cfg.py" $?
  cmsRun ${LOCAL_TEST_DIR}/testGetProductAtEndUnscheduled_cfg.py || die "cmsRun testGetProductAtEndUnscheduled_cfg.py" $?

exit 0
