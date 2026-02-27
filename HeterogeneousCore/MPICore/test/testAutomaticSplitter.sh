#! /bin/bash

# Shell script for testing the automattic config splitter for MPI in CMSSW

SPLITTER=$(realpath "$1")
WHOLE_CONFIG=$(realpath "$2")

LOCAL_PATH="./autosplit_result/local_test_config.py"
REMOTE_PATH="./autosplit_result/remote_test_config.py"

python3 "$SPLITTER" "$WHOLE_CONFIG" \
  --remote-modules triggerEventProducer testReadTriggerResults rawDataBufferProducer testReadFEDRawDataCollection \
  -ol "$LOCAL_PATH" -or "$REMOTE_PATH"

"$CMSSW_BASE"/src/HeterogeneousCore/MPICore/test/testMPICommWorld.sh "$LOCAL_PATH" "$REMOTE_PATH"
