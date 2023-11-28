#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

# list basic test
cmsRun ${SCRAM_TEST_PATH}/Py8_bplus_evtgen1_cfg.py || die 'Failure using Py8_bplus_evtgen1_cfg.py' $?

# extra test
cmsRun ${SCRAM_TEST_PATH}/../plugins/test/Py8_lambadb_evtgen1_Lb2plnuLCSR_cfg.py || die 'Py8_lambadb_evtgen1_Lb2plnuLCSR_cfg.py' $?
