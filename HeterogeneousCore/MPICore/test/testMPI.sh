#! /bin/bash
# Shell script for testing CMSSW over MPI

mkdir -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test
DIR=$(mktemp -d -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test)
echo "Running MPI tests at $DIR/"
pushd $DIR > /dev/null

# start an MPI server to let independent CMSSW processes find each other
echo "Starting the Open RTE data server"
ompi-server -r server.uri -d >& ompi-server.log &
SERVER_PID=$!
disown
# wait until the ORTE server logs 'up and running'
while ! grep -q 'up and running' ompi-server.log; do
  sleep 1s
done

# Note: "mpirun --mca pmix_server_uri file:server.uri" is required to make the
# tests work inside a singularity/apptainer container. Without a container the
# cmsRun commands can be used directly.

# start the "follower" CMSSW job(s)
{
  mpirun --mca pmix_server_uri file:server.uri -n 1 -- cmsRun $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/testMPIFollower.py >& mpifollower.log
  echo $? > mpifollower.status
} &
FOLLOWER_PID=$!

# wait to make sure the MPISource has established the connection to the ORTE server
sleep 3s

# start the "controller" CMSSW job(s)
{
  mpirun --mca pmix_server_uri file:server.uri -n 1 -- cmsRun $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/testMPIController.py >& mpicontroller.log
  echo $? > mpicontroller.status
} &
CONTROLLER_PID=$!

# wait for the CMSSW jobs to finish
wait $CONTROLLER_PID $FOLLOWER_PID

# print the jobs' output and check the jobs' exit status
echo '========== testMPIController ==========='
cat mpicontroller.log
MPICONTROLLER_STATUS=$(< mpicontroller.status)
echo '========================================'
echo
echo '=========== testMPIFollower ============'
cat mpifollower.log
MPISOURCE_STATUS=$(< mpifollower.status)
echo '========================================'

# stop the MPI server and cleanup the URI file
kill $SERVER_PID

popd > /dev/null
exit $((MPISOURCE_STATUS > MPICONTROLLER_STATUS ? MPISOURCE_STATUS : MPICONTROLLER_STATUS))
