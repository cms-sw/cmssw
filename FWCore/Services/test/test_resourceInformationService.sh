#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun -p ${SCRAM_TEST_PATH}/test_resourceInformationService_cfg.py &> test_resourceInformationService.log || die "cmsRun test_resourceInformationService_cfg.py" $?

grep -A 1 "acceleratorTypes:" test_resourceInformationService.log | grep "GPU" || die "Check for existence of GPU acceleratorType" $?
grep -A 1 "cpu models:" test_resourceInformationService.log | grep "None" && die "Check there is at least one model (not None)" 1

exit 0
