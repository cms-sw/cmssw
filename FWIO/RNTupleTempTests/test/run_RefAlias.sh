#!/bin/bash
test=ref_alias_compare_

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  echo ${test}drop_alias_cfg.py ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}drop_alias_cfg.py || die "cmsRun ${test}drop_alias_cfg.py" $?

  echo ${test}drop_original_cfg.py ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}drop_original_cfg.py || die "cmsRun ${test}drop_original_cfg.py" $?

  echo ${test}read_alias_cfg.py------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}read_alias_cfg.py || die "cmsRun ${test}read_alias_cfg.py" $?

  echo ${test}read_original_cfg.py------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}read_original_cfg.py || die "cmsRun ${test}read_original_cfg.py" $?

exit 0
