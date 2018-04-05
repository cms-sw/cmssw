#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

F1=${LOCAL_TEST_DIR}/testMerge_cfg.py
F2=${LOCAL_TEST_DIR}/testNoMerge_cfg.py


(cmsRun $F1) > test_merge.log || die "Failure using $F1" $?

diff ${LOCAL_TEST_DIR}/testMerge.log test_merge.log || die "comparing test_merge.log" $?


(cmsRun $F2) > test_no_merge.log || die "Failure using $F2" $?
diff ${LOCAL_TEST_DIR}/testNoMerge.log test_no_merge.log || die "comparing test_no_merge.log" $?

popd