#!/bin/bash
test=ref_alias_compare_

function die { echo Failure $1: status $2 ; exit $2 ; }

echo LOCAL_TMP_DIR = ${LOCAL_TMP_DIR}

pushd ${LOCAL_TMP_DIR}
  echo ${test}drop_alias_cfg.py ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}drop_alias_cfg.py || die "cmsRun ${test}drop_alias_cfg.py" $?

  echo ${test}drop_original_cfg.py ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}drop_original_cfg.py || die "cmsRun ${test}drop_original_cfg.py" $?

  echo ${test}read_alias_cfg.py------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}read_alias_cfg.py || die "cmsRun ${test}read_alias_cfg.py" $?

  echo ${test}read_original_cfg.py------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}read_original_cfg.py || die "cmsRun ${test}read_original_cfg.py" $?

popd

exit 0
