 #!/bin/bash -ex
TEST_DIR=$CMSSW_BASE/src/RecoPPS/Local/test
echo "test dir: $TEST_DIR"

cmsRun ${TEST_DIR}/2023_lhcinfo_test_recoCTPPS_cfg.py
