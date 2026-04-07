#! /bin/bash

# Shell script for testing the automattic config splitter for MPI in CMSSW

WHOLE_CONFIG=$(realpath "$1")

LOCAL_PATH="./autosplit_result/local_test_config.py"
REMOTE_PATH_1="./autosplit_result/remote_test_config_1.py"
REMOTE_PATH_2="./autosplit_result/remote_test_config_2.py"

edmMpiSplitConfig "$WHOLE_CONFIG" -l "$LOCAL_PATH"\
  --remote-modules triggerEventProducer testReadTriggerResults -r "$REMOTE_PATH_1" :\
  --remote-modules rawDataBufferProducer testReadFEDRawDataCollection -r "$REMOTE_PATH_2"


"$CMSSW_BASE"/src/HeterogeneousCore/MPICore/test/testMPICommWorld.sh "$LOCAL_PATH" "$REMOTE_PATH_1" "$REMOTE_PATH_2" 
