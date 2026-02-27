#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun ${SCRAM_TEST_PATH}/test_resourceInformationService_cfg.py &> test_resourceInformationService.log || die "cmsRun test_resourceInformationService_cfg.py" $?

grep -A 4 "selectedAccelerators:" test_resourceInformationService.log | grep "cpu" || die "Check for existence of cpu in selectedAccelerators" $?
grep -A 4 "selectedAccelerators:" test_resourceInformationService.log | grep "gpu-foo" || die "Check for existence of gpu-foo in selectedAccelerators" $?
grep -A 4 "selectedAccelerators:" test_resourceInformationService.log | grep "test1" || die "Check for existence of test1 in selectedAccelerators" $?
grep -A 4 "selectedAccelerators:" test_resourceInformationService.log | grep "test2" || die "Check for existence of test2 in selectedAccelerators" $?
grep -A 1 "cpu models:" test_resourceInformationService.log | grep "None" && die "Check there is at least one model (not None)" 1

exit 0
