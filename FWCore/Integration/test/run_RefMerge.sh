#!/bin/bash

test=ref_merge_

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
  echo ${test}prod1 ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --extraProducers --fileName 'ref_merge_prod1.root' || die "cmsRun ${test}prod_cfg.py --extraProducers" $?

  echo ${test}prod2 ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --firstLumi 10 --fileName 'ref_merge_prod2.root'|| die "cmsRun ${test}prod_cfg.py" $?

  echo ${test}MERGE------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile1 'ref_merge_prod1.root' --inFile2 'ref_merge_prod2.root' --outFile 'ref_merge.root' || die "cmsRun ${test}cfg.py" $?


  echo ${test}MERGE promptRead------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --fileName 'ref_merge.root'  --promptRead || die "cmsRun ${test}test_cfg.py" $?

  echo ${test}keepAllProd ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}prod_cfg.py --extraProducers --keepAllProducts --fileName 'ref_merge_prod_all.root' || die "cmsRun ${test}prod_cfg.py --keepAllProducts" $?

  echo ${test}MERGE_keepAll1st ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}cfg.py --inFile1 'ref_merge_prod_all.root' --inFile2 'ref_merge_prod2.root' --outFile 'ref_merge_all1st.root' || die "cmsRun ${test}cfg.py" $?

  echo ${test}test_all1st------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${test}test_cfg.py --fileName 'ref_merge_all1st.root' || die "cmsRun ${test}test_cfg.py all1st" $?

  #note having all be the second file does not work as PoolSource enforces that subsequent files must have a strict subset
  # of the branches in the first file read

exit 0
