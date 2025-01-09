#!/bin/bash

test=ref_merge_

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
#------------- same configs, same run ------------

  echo ${test}prod_a ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --fileName 'ref_merge_proda.root' || die "cmsRun ${test}prod_cfg.py" $?

  echo ${test}prod_b ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --firstLumi 10 --fileName 'ref_merge_prodb.root'|| die "cmsRun ${test}prod_cfg.py" $?

#------------- same configs, same run using cat ------------
cat ref_merge_proda.root ref_merge_prodb.root > ref_merge_cat.root

  echo ${test}test_cat------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --fileName 'ref_merge_cat.root' && die "cmsRun ${test}test_cfg.py same" 1

#------------- same configs different stored products, same run ------------

  echo ${test}prod_ass ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --firstLumi 10 --fileName 'ref_merge_prod_all.root' --keepAllProducts || die "cmsRun ${test}prod_cfg.py" $?

  echo ${test}MERGE_diff_prods2------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile1 'ref_merge_proda.root' --inFile2 'ref_merge_prod_all.root' --outFile 'ref_merge_diff_prods2.root' && die "cmsRun ${test}cfg.py diff prods 2" 1

#------------- different configs ------------

  echo ${test}prod1 ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --extraProducers --fileName 'ref_merge_prod1.root' || die "cmsRun ${test}prod_cfg.py --extraProducers" $?

  echo ${test}prod2 ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --firstLumi 10 --fileName 'ref_merge_prod2.root'|| die "cmsRun ${test}prod_cfg.py" $?

  echo ${test}MERGE_diff_configs------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile1 'ref_merge_prod1.root' --inFile2 'ref_merge_prod2.root' --outFile 'ref_merge.root' && die "cmsRun ${test}cfg.py diff configs" 1

#------------- different configs and different products ------------

  echo ${test}keepAllProd ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --extraProducers --keepAllProducts --fileName 'ref_merge_prod_all.root' || die "cmsRun ${test}prod_cfg.py --keepAllProducts" $?

  echo ${test}MERGE_keepAll1st ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile2 'ref_merge_prod_all.root' --inFile1 'ref_merge_prod2.root' --outFile 'ref_merge_all1st.root' && die "cmsRun ${test}cfg.py" 1

exit 0
