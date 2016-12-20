#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun ${LOCAL_TEST_DIR}/streamGrapher_make_file_to_read_cfg.py || die "cmsRun streamGrapher_make_file_to_read_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/streamGrapher_tracer_cfg.py > raw_tracer_out.log 2>&1 || die "cmsRun streamGrapher_tracer_cfg.py" $?
grep '+' raw_tracer_out.log > tracer_output.log
${LOCAL_TEST_DIR}/../bin/edmStreamStallGrapher.py tracer_output.log || die "edmStreamStallGrapher.py tracer_output.log " $?
