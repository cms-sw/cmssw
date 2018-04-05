#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun ${LOCAL_TEST_DIR}/testGetProductAtEnd_cfg.py || die "cmsRun testGetProductAtEnd_cfg.py" $?
  cmsRun ${LOCAL_TEST_DIR}/testGetProductAtEndUnscheduled_cfg.py || die "cmsRun testGetProductAtEndUnscheduled_cfg.py" $?

popd

exit 0
