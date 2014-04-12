#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun ${LOCAL_TEST_DIR}/testEdmConfigDump_cfg.py > testEdmConfigDump.log || die "cmsRun testEdmConfigDump_cfg.py" $?
  edmProvDump testEdmProvDump.root > provdump.log
  python ${LOCAL_TEST_DIR}/removeChangingParts.py provdump.log > provdump1.log
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/provdump.log provdump1.log  || die "comparing provdump.log" $?

popd

exit 0
