#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/testPrimaryVertexFilter_cfg.py || die 'Failure using testPrimaryVertexFilter' $?

cmsRun ${LOCAL_TEST_DIR}/testPrimaryVertexObjectFilter_cfg.py || die 'Failure using testPrimaryVertexObjectFilter' $?

cmsRun ${LOCAL_TEST_DIR}/cmsswWithPythonConfig_cfg.py || die 'Failure using cmsswWithPythonConfig' $?

FWLiteWithBasicAnalyzer ${LOCAL_TEST_DIR}/fwliteWithPythonConfig_cfg.py || die 'Failure using fwliteWithPythonConfig' $?

