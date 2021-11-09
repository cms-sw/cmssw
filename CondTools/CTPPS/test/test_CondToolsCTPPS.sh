 #!/bin/bash -ex
TEST_DIR=$CMSSW_BASE/src/CondTools/CTPPS/test
echo "test dir: $TEST_DIR"

cmsRun ${TEST_DIR}/write_PPSAssociationCuts_cfg.py
