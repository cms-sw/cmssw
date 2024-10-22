#!/bin/sh

function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

(cmsRun ${SCRAM_TEST_PATH}/dumpMkFitGeometry.py) || die "failed to run dumpMkFitGeometry.py" $?
(cmsRun ${SCRAM_TEST_PATH}/dumpMkFitGeometryPhase2.py) || die "failed to run dumpMkFitGeometryPhase2.py" $?
