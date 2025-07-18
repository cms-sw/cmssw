#!/bin/bash -ex
TEST_DIR=$CMSSW_BASE/src/RecoPPS/Local/test
echo "test dir: $TEST_DIR"

# test quickly 50 events (with strips) to see if reco is not crashing
cmsRun ${TEST_DIR}/global_reco_PPS_test_cfg.py maxEvents=50 strips=1
