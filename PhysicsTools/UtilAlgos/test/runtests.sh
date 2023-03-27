#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/testPrimaryVertexFilter_cfg.py || die 'Failure using testPrimaryVertexFilter' $?

cmsRun ${SCRAM_TEST_PATH}/testPrimaryVertexObjectFilter_cfg.py || die 'Failure using testPrimaryVertexObjectFilter' $?

cmsRun ${SCRAM_TEST_PATH}/cmsswWithPythonConfig_cfg.py || die 'Failure using cmsswWithPythonConfig' $?

FWLiteWithBasicAnalyzer ${SCRAM_TEST_PATH}/fwliteWithPythonConfig_cfg.py || die 'Failure using fwliteWithPythonConfig' $?

