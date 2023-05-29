#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun -e ${SCRAM_TEST_PATH}/test_Timing_cfg.py &> test_Timing.log || die "cmsRun test_Timing_cfg.py" $?

NAMES=("AvgEventTime" "EventThroughput" "MaxEventTime" "MinEventTime" "NumberOfStreams" "NumberOfThreads" "TotalEventSetupTime" \
	 "TotalInitCPU" "TotalInitTime" "TotalJobCPU" "TotalJobChildrenCPU" "TotalJobTime" "TotalLoopCPU" "TotalLoopTime" "TotalNonModuleTime")

grep "Time Summary" test_Timing.log || die "Check for 'Time Summary' message" $?
grep '<PerformanceSummary Metric="Timing">' FrameworkJobReport.xml || die "Check for Timing group in FJR" $?

for NAME in ${NAMES[@]}; do
  grep $NAME FrameworkJobReport.xml || die "Check for $NAME in FJR" $?
done
exit 0
