#!/bin/sh

function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

(cmsRun ${LOCAL_TEST_DIR}/dumpMkFitGeometry.py) || die "failed to run dumpMkFitGeometry.py" $?
