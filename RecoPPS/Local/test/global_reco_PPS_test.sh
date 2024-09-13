 #!/bin/bash -ex
TEST_DIR=$CMSSW_BASE/src/RecoPPS/Local/test
echo "test dir: $TEST_DIR"

# run quickly 50 events to check if reco process is not crashing
cmsRun ${TEST_DIR}/global_reco_PPS_test_cfg.py maxEvents=50
