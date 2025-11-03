#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun -e ${SCRAM_TEST_PATH}/tracer_cfg.py &> test_Timing.log || die "cmsRun tracer_cfg.py" $?
edmTracerCompactLogViewer.py -j tracer.log &> tracer.json || die "edmTracerCompactLogViewer.py" $?
grep -q '"type": 24' tracer.json || die "Check for processPython transition in JSON" $?
grep -q '"type": 23' tracer.json || die "Check for startServices transition in JSON" $?
grep -q '"type": 22' tracer.json || die "Check for constructESModules transition in JSON" $?
grep -q '"type": 21' tracer.json || die "Check for finishSchedule transition in JSON" $?
grep -q '"type": 20' tracer.json || die "Check for createRunLumiEvents transition in JSON" $?
grep -q '"type": 19' tracer.json || die "Check for scheduleConsistencyCheck transition in JSON" $?
grep -q '"type": 18' tracer.json || die "Check for finalizeEventSetupConfiguration transition in JSON" $?
grep -q '"type": 17' tracer.json || die "Check for finalizeEDModules transition in JSON" $?
grep -q '"type": 16' tracer.json || die "Check for construction transition in JSON" $?
#grep -q '"type": -16' tracer.json || die "Check for destruction transition in JSON" $? #there are no destruction calls in the file
grep -q '"type": 12' tracer.json || die "Check for beginJob transition in JSON" $?
grep -q '"type": -12' tracer.json || die "Check for endJob transition in JSON" $?
grep -q '"type": 11' tracer.json || die "Check for beginStream transition in JSON" $?
grep -q '"type": -11' tracer.json || die "Check for endStream transition in JSON" $?
#grep -q '"type": 10' tracer.json || die "Check for open file transition in JSON" $? #no file in EmptySource
grep -q '"type": -10' tracer.json || die "Check for write file transition in JSON" $?
grep -q '"type": 9' tracer.json || die "Check for beginProcessBlock transition in JSON" $?
grep -q '"type": -9' tracer.json || die "Check for endProcessBlock transition in JSON" $?
#grep -q '"type": 8' tracer.json || die "Check for inputProcessBlock transition in JSON" $? #no file transition for inputProcessBlock in this test
grep -q '"type": -8' tracer.json && die "Check for missing -8 transition in JSON" $?
grep -q '"type": 7' tracer.json && die "Check for missing 7 transition in JSON" $?
grep -q '"type": -7' tracer.json || die "Check for globalWriteRun transition in JSON" $?
grep -q '"type": 6' tracer.json || die "Check for globalBeginRun transition in JSON" $?
grep -q '"type": -6' tracer.json || die "Check for globalEndRun transition in JSON" $?
grep -q '"type": 5' tracer.json || die "Check for streamBeginRun transition in JSON" $?
grep -q '"type": -5' tracer.json || die "Check for streamEndRun transition in JSON" $?
grep -q '"type": 4' tracer.json && die "Check for missing 4 transition in JSON" $?
grep -q '"type": -4' tracer.json || die "Check for globalWriteLumi transition in JSON" $?
grep -q '"type": 3' tracer.json || die "Check for globalBeginLumi transition in JSON" $?
grep -q '"type": -3' tracer.json || die "Check for globalEndLumi transition in JSON" $?
grep -q '"type": 2' tracer.json || die "Check for streamBeginLumi transition in JSON" $?
grep -q '"type": -2' tracer.json || die "Check for streamEndLumi transition in JSON" $?
grep -q '"type": 1' tracer.json && die "Check for missing 1 transition in JSON" $?
grep -q '"type": -1' tracer.json || die "Check for clearEvent transition in JSON" $?
grep -q '"type": 0' tracer.json || die "Check for event transition in JSON" $?