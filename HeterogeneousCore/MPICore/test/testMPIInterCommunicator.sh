#! /bin/bash
# Shell script for testing CMSSW over MPI
CONTROLLER=$(realpath $1)
FOLLOWER=$(realpath $2)

# Make sure the CMSSW environment has been loaded.
if [ -z "$CMSSW_BASE" ]; then
  eval `scram runtime -sh`
fi

# The CM pml leads to silent communication failures on some machines.
# Until this is understood and fixed, keep it disabled.
export OMPI_MCA_pml='^cm'

mkdir -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test
DIR=$(mktemp -d -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test)
echo "Running MPI tests at $DIR/"
pushd $DIR > /dev/null

# Start an MPI server to let independent CMSSW processes find each other.
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

# Start the "follower" CMSSW job(s).
{
  mpirun --mca pmix_server_uri file:server.uri -n 1 -- cmsRun $FOLLOWER >& follower.log
  echo $? > follower.status
} &
FOLLOWER_PID=$!

# Wait until the MPISource has established the connection to the ORTE server.
while ! grep -q 'waiting for a connection to the MPI server' follower.log; do
  sleep 1s
done

# Start the "controller" CMSSW job(s).
{
  mpirun --mca pmix_server_uri file:server.uri -n 1 -- cmsRun $CONTROLLER >& controller.log
  echo $? > controller.status
} &
CONTROLLER_PID=$!

# Wait for the CMSSW jobs to finish.
wait $CONTROLLER_PID $FOLLOWER_PID

# Print the jobs' output and check the jobs' exit status.
echo '========== testMPIController ==========='
cat controller.log
MPICONTROLLER_STATUS=$(< controller.status)
echo '========================================'
echo
echo '=========== testMPIFollower ============'
cat follower.log
MPISOURCE_STATUS=$(< follower.status)
echo '========================================'

# Stop the MPI server and cleanup the URI file.
kill $SERVER_PID

popd > /dev/null
exit $((MPISOURCE_STATUS > MPICONTROLLER_STATUS ? MPISOURCE_STATUS : MPICONTROLLER_STATUS))
