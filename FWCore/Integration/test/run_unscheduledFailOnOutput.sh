#!/bin/bash

test=unscheduled_fail_on_output_
CFG_DIR=${LOCAL_TEST_DIR}/../python/test

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun ${CFG_DIR}/${test}Rethrow_cfg.py && die "cmsRun ${test}Rethrow_cfg.py did not fail" $?

  cmsRun ${CFG_DIR}/${test}IgnoreCompletely_cfg.py || die "cmsRun ${test}IgnoreCompletely_cfg.py" $?
  cmsRun ${CFG_DIR}/${test}read_found_events.py || die "cmsRun ${test}read_found_events.py" $?

  cmsRun -p ${CFG_DIR}/${test}FailModule_cfg.py || die "cmsRun ${test}FailModule_cfg.py" $?
  cmsRun ${CFG_DIR}/${test}read_found_events.py || die "cmsRun ${test}read_found_events.py" $?

  cmsRun -p ${CFG_DIR}/${test}FailPath_cfg.py || die "cmsRun ${test}FailPath_cfg.py" $?
  #NOTE: Following fails because of an assert, plus we don't know what behavior we actually want in this case
  cmsRun ${CFG_DIR}/${test}read_no_events.py || die "cmsRun ${test}read_no_events.py" $?

  cmsRun -p ${CFG_DIR}/${test}SkipEvent_cfg.py || die "cmsRun ${test}SkipEvent_cfg.py" $?
  #NOTE: Following fails because of an assert, plus we don't know what behavior we actually want in this case
  cmsRun ${CFG_DIR}/${test}read_no_events.py || die "cmsRun ${test}read_no_events.py" $?

popd

exit 0
