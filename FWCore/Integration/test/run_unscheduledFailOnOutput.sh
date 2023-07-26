#!/bin/bash

test=unscheduled_fail_on_output_
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
CFG_DIR=${LOCAL_TEST_DIR}/../python/test

function die { echo Failure $1: status $2 ; exit $2 ; }

  cmsRun ${CFG_DIR}/${test}Rethrow_cfg.py && die "cmsRun ${test}Rethrow_cfg.py did not fail" 1

  cmsRun ${CFG_DIR}/${test}IgnoreCompletely_cfg.py || die "cmsRun ${test}IgnoreCompletely_cfg.py" $?
  cmsRun ${CFG_DIR}/${test}read_found_events.py || die "cmsRun ${test}read_found_events.py for IgnoreCompletely" $?

  cmsRun -p ${CFG_DIR}/${test}TryToContinue_cfg.py || die "cmsRun ${test}TryToComplete_cfg.py" $?
  cmsRun ${CFG_DIR}/${test}read_no_events.py || die "cmsRun ${test}read_no_events.py" $?

  cmsRun -p ${CFG_DIR}/${test}no_dependency_TryToContinue_cfg.py || die "cmsRun ${test}no_dependency_TryToComplete_cfg.py" $?
  cmsRun ${CFG_DIR}/${test}read_found_events.py || die "cmsRun ${test}read_found_events.py for TryToComplete" $?

exit 0
