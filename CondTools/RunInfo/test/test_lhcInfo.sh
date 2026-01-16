#!/bin/sh

LOCAL_TEST_DIR=$CMSSW_BASE/src/CondTools/RunInfo/test

# Source shared utility functions
source "${LOCAL_TEST_DIR}/testing_utils.sh"

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerFillWriter_cfg.py || die "cmsRun LHCInfoPerFillWriter_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerFillTester_cfg.py || die "cmsRun LHCInfoPerFillTester_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerLSWriter_cfg.py   || die "cmsRun LHCInfoPerLSWriter_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerLSTester_cfg.py || die "cmsRun LHCInfoPerLSTester_cfg.py" $?
