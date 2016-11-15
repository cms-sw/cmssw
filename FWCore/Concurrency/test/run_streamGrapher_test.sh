#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun ${LOCAL_TEST_DIR}/streamGrapher_make_file_to_read_cfg.py || die "cmsRun streamGrapher_make_file_to_read_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/streamGrapher_stallMonitor_cfg.py || die "cmsRun streamGrapher_stallMonitor_cfg.py" $?
${LOCAL_TEST_DIR}/../bin/edmStreamStallGrapher.py stallMonitor.log || die "edmStreamStallGrapher.py stallMonitor.log " $?
