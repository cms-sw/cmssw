#!/bin/bash

test=testParameterSet

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

# The process in the first test below should fail.  The test
# passes when the process fails because of an illegal parameter.
  echo ${test} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}_cfg.py 2> ${test}.txt
  grep "Illegal parameter" ${test}.txt || die "cmsRun ${test}_cfg.py" $?

# Auto generate a cfi file
  echo edmWriteConfigs ------------------------------------------------------------
  edmWriteConfigs pluginTestProducerWithPSetDesc.so || die "edmWriteConfigs pluginTestProducerWithPSetDesc.so" $?

# Make sure we can run using the cfi file generated in the previous process
  echo cmsRun runAutoProducedCfi_cfg.py ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/runAutoProducedCfi_cfg.py || die "cmsRun runAutoProducedCfi_cfg.py" $?

# Compare the cfi file to a reference file to ensure it is correct
  diff ${LOCAL_TMP_DIR}/testProducerWithPsetDesc_cfi.py ${LOCAL_TEST_DIR}/unit_test_outputs/testProducerWithPsetDesc_cfi.py || die "comparing testProducerWithPsetDesc_cfi.py" $?

popd

exit 0
