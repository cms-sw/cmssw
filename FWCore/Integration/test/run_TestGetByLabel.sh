#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

TEST_NAME=TestGetByLabel
if [ -d ${TEST_NAME} ]; then
    rm -fR ${TEST_NAME}
fi
mkdir ${TEST_NAME}

pushd ${TEST_NAME}

TEST_PATH=${LOCALTOP}/src/FWCore/Integration/test

cmsRun ${TEST_PATH}/testGetByLabelStep1_cfg.py || die "Failed cmsRun testGetByLabel_step1_cfg.py" $1

cmsRun ${TEST_PATH}/testGetByLabelStep2_cfg.py || die "Failed cmsRun testGetByLabel_step2_cfg.py" $1
cmsRun ${TEST_PATH}/testGetByLabelStep2_cfg.py -- --noConsumes || die "Failed cmsRun testGetByLabel_step2_cfg.py --noConsumes" $1
cmsRun ${TEST_PATH}/testGetByLabelStep2_cfg.py -- --thing || die "Failed cmsRun testGetByLabel_step2_cfg.py --thing" $1
cmsRun ${TEST_PATH}/testGetByLabelStep2_cfg.py -- --thing --noConsumes || die "Failed cmsRun testGetByLabel_step2_cfg.py --thing --noConsumes" $1
cmsRun ${TEST_PATH}/testGetByLabelStep2_cfg.py -- --otherInt || die "Failed cmsRun testGetByLabel_step2_cfg.py --otherInt" $1
cmsRun ${TEST_PATH}/testGetByLabelStep2_cfg.py -- --otherInt --noConsumes || die "Failed cmsRun testGetByLabel_step2_cfg.py --otherInt --noConsumes" $1

popd
