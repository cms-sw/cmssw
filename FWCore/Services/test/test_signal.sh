#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun -e ${SCRAM_TEST_PATH}/test_signal_cfg.py &> test_signal.log || die "cmsRun test_signal_cfg.py" $?

grep "%MSG-s ShutdownSignal:" test_signal.log || die "Check for shutdown signal message" $?
grep "<ShutdownSignal/>" FrameworkJobReport.xml || die "Check for signal element in FJR" $?

exit 0
