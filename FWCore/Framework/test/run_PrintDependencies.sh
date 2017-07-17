#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/testPrintDependencies.py

cmsRun $F1 2>&1 | grep "depends on data products" >& ${LOCAL_TEST_DIR}/dependencies.txt || die "Failure using $F1" $?
diff -q ${LOCAL_TEST_DIR}/dependencies.txt ${LOCAL_TEST_DIR}/unit_test_outputs/dependencies.txt || die "dependencies differ" $?

rm -f ${LOCAL_TEST_DIR}/dependencies.txt

