#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
cmsRun ${LOCAL_TEST_DIR}/streamGrapher_make_file_to_read_cfg.py || die "cmsRun streamGrapher_make_file_to_read_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/streamGrapher_stallMonitor_cfg.py > raw_output.log 2>&1 || die "cmsRun streamGrapher_stallMonitor_cfg.py" $?
grep '+' raw_output.log > output_from_tracer.log
${LOCAL_TEST_DIR}/../scripts/edmStreamStallGrapher.py output_from_tracer.log || die "edmStreamStallGrapher.py output_from_tracer.log " $?
${LOCAL_TEST_DIR}/../scripts/edmStreamStallGrapher.py stallMonitor.log || die "edmStreamStallGrapher.py stallMonitor.log " $?
