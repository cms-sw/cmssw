#!/bin/bash

test=ref_merge_

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
#------------- same configs, same run ------------

  echo ${test}prod_a ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --fileName 'ref_merge_proda.root' || die "cmsRun ${test}prod_cfg.py" $?

  echo ${test}prod_b ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --firstLumi 10 --fileName 'ref_merge_prodb.root'|| die "cmsRun ${test}prod_cfg.py" $?

  echo ${test}MERGE_same------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile1 'ref_merge_proda.root' --inFile2 'ref_merge_prodb.root' --outFile 'ref_merge_same.root' || die "cmsRun ${test}cfg.py same" $?

  echo ${test}test_same------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --fileName 'ref_merge_same.root' || die "cmsRun ${test}test_cfg.py same" $?

#------------- same configs different stored products, same run ------------
# works if subsequent files have a strict subset of stored products of the first file

  echo ${test}prod_b ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --firstLumi 10 --fileName 'ref_merge_prod_all.root' --keepAllProducts || die "cmsRun ${test}prod_cfg.py" $?

  echo ${test}MERGE_diff_prods1------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile2 'ref_merge_proda.root' --inFile1 'ref_merge_prod_all.root' --outFile 'ref_merge_diff_prods.root' || die "cmsRun ${test}cfg.py diff prods" $?

  echo ${test}test_diff_prods1------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --fileName 'ref_merge_diff_prods.root' || die "cmsRun ${test}test_cfg.py diff prods" $?

#------------- same configs, different run ------------

  echo ${test}prod_run10 ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --firstRun 10 --fileName 'ref_merge_prod_run10.root'|| die "cmsRun ${test}prod_cfg.py run10" $?

  echo ${test}MERGE_diff_runs------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile1 'ref_merge_proda.root' --inFile2 'ref_merge_prod_run10.root' --outFile 'ref_merge_diffRuns.root' || die "cmsRun ${test}cfg.py diff runs" $?

  echo ${test}test_diff_runs------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --fileName 'ref_merge_diffRuns.root' || die "cmsRun ${test}test_cfg.py diff runs" $?

exit 0
