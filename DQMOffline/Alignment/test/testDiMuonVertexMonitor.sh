#!/bin/bash

 function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

cmsRun ${LOCAL_TEST_DIR}/DiMuonVertexValidator_cfg.py  || die "Failure using DiMuonVertexValidator_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/DiMuonVertex_HARVESTING.py || die "Failure using DiMuonVertex_HARVESTING.py" $? 
