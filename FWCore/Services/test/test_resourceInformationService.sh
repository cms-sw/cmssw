#!/bin/bash

LOCAL_TEST_DIR=${CMSSW_BASE}/src/FWCore/Services/test
LOCAL_TMP_DIR=${CMSSW_BASE}/tmp/${SCRAM_ARCH}

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

cmsRun -p ${LOCAL_TEST_DIR}/test_resourceInformationService_cfg.py &> test_resourceInformationService.log || die "cmsRun test_resourceInformationService_cfg.py" $?

grep -A 1 "acceleratorTypes:" test_resourceInformationService.log | grep "GPU" || die "Check for existence of GPU acceleratorType" $?
grep -A 1 "cpu models:" test_resourceInformationService.log | grep "None" && die "Check there is at least one model (not None)" 1
popd

exit 0
