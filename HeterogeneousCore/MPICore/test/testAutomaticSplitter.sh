#! /bin/bash

# Shell script for testing the automattic config splitter for MPI in CMSSW

WHOLE_CONFIG=$(realpath "$1")

LOCAL_PATH="./autosplit_result/local_test_config.py"
REMOTE_PATH="./autosplit_result/remote_test_config.py"

edmMpiSplitConfig "$WHOLE_CONFIG" \
  --remote-modules triggerEventProducer testReadTriggerResults rawDataBufferProducer testReadFEDRawDataCollection \
  -l "$LOCAL_PATH" -r "$REMOTE_PATH"

"$CMSSW_BASE"/src/HeterogeneousCore/MPICore/test/testMPICommWorld.sh "$LOCAL_PATH" "$REMOTE_PATH"
