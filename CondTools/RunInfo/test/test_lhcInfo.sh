#!/bin/sh

LOCAL_TEST_DIR=$CMSSW_BASE/src/CondTools/RunInfo/test

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerFillWriter_cfg.py || die "cmsRun LHCInfoPerFillWriter_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerFillTester_cfg.py || die "cmsRun LHCInfoPerFillTester_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerLSWriter_cfg.py   || die "cmsRun LHCInfoPerLSWriter_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerLSTester_cfg.py || die "cmsRun LHCInfoPerLSTester_cfg.py" $?
