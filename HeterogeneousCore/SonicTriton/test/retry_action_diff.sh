#!/bin/bash
LOCALTOP=$1
shift
# any further arguments are passed to cmsRun (e.g. --threads 4 --streams 4)

# Sleep durations in seconds, can be overridden from the environment
SERVER_WAIT=${SERVER_WAIT:-60}            # allow the servers to initialize
CLIENT_WAIT=${CLIENT_WAIT:-45}            # allow the client to run before killing the first server
SECOND_KILL_WAIT=${SECOND_KILL_WAIT:-10}  # allow the client to run before killing the second server

# Stop the server container instance(s) on exit, including early exits.
# cmsTriton runs tritonserver inside an apptainer instance: the "kill" below only stops the tritonserver
# process (which is what makes the client retry), while the instance itself keeps running until
# "cmsTriton stop" is called, even when the test works as intended.
cleanup() {
	cmsTriton -n server1 stop
	cmsTriton -n server2 stop
}
trap cleanup EXIT

# Start the server
cmsTriton -v -P 8010 -n server1 -f -L -m $(edmFileInPath HeterogeneousCore/SonicTriton/data/models/gat_test/config.pbtxt) start &
cmsTriton -v -P 8020 -n server2 -f -L -m $(edmFileInPath HeterogeneousCore/SonicTriton/data/models/gat_test/config.pbtxt) start &

# Sleep to allow the server to initialize
sleep ${SERVER_WAIT}

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
cmsRun ${LOCALTOP}/src/HeterogeneousCore/SonicTriton/test/tritonTest_cfg.py --maxEvents 100 --modules TritonIdentityProducer --models ragged_io  --address server1 0.0.0.0 8011 --address server2 0.0.0.0 8021 --retryAction diff --verbose "$@" &

# Allow the client some time to run
sleep ${CLIENT_WAIT}

# Kill the first server
echo "Killing server with PID $SERVER1_PID"
kill $SERVER1_PID

# Allow the client some time to run
sleep ${SECOND_KILL_WAIT}

echo "Killing server with PID $SERVER2_PID"
kill $SERVER2_PID
# Wait for client process to complete
wait
echo "Client process completed."
