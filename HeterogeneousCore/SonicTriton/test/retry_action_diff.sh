#!/bin/bash
LOCALTOP=$1

# Start the server
cmsTriton -v -P 8010 -n server1 -f -L -m /cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_15_1_0_pre6/external/el9_amd64_gcc12/data/HeterogeneousCore/SonicTriton/data/models/gat_test/config.pbtxt start &
cmsTriton -v -P 8020 -n server2 -f -L -m /cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_15_1_0_pre6/external/el9_amd64_gcc12/data/HeterogeneousCore/SonicTriton/data/models/gat_test/config.pbtxt start &

# Sleep to allow the server to initialize
sleep 60

# Get the true PID of the server
SERVER1_PID=$(ps -e -o pid,cmd | grep "tritonserver" | grep "8010" | grep -v "grep" | awk '{print $1}')
SERVER2_PID=$(ps -e -o pid,cmd | grep "tritonserver" | grep "8020" | grep -v "grep" | awk '{print $1}')

echo "Servers started with  $SERVER1_PID and $SERVER2_PID"
ps aux | grep tritonserver

if [ -z "$SERVER1_PID" ] || [ -z "$SERVER2_PID" ]; then
    echo "Server process could not be started."
    exit 1
fi

# Start the client
cmsRun ${LOCALTOP}/src/HeterogeneousCore/SonicTriton/test/tritonTest_cfg.py --maxEvents 100 --modules TritonIdentityProducer --models ragged_io  --address server1 0.0.0.0 8011 --address server2 0.0.0.0 8021 --retryAction diff --verbose &

# Allow the client some time to run
sleep 60

# Kill the first server
echo "Killing server with PID $SERVER1_PID"
kill $SERVER1_PID

# Allow the client some time to run
sleep 30

echo "Killing server with PID $SERVER2_PID"
kill $SERVER2_PID
# Wait for client process to complete
wait
echo "Client process completed."
