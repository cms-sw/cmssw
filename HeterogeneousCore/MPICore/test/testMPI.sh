#! /bin/bash
# Shell script for testing CMSSW over MPI

mkdir -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test
DIR=$(mktemp -d -p $CMSSW_BASE/tmp/$SCRAM_ARCH/test)
echo "Running MPI tests at $DIR/"
pushd $DIR > /dev/null

# start an MPI server to let independent CMSSW processes find each other
ompi-server -r server.uri -d &> ompi-server.log &
SERVER_PID=$!
disown

# create a test file
cmsRun $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/createTestFile.py &> testfile.log

# start the "follower" CMSSW job(s)
{
  #mpirun --mca pmix_server_uri file:server.uri -n 1 -- ...
  cmsRun $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/testMPISource.py &> mpisource.log
  echo $? > mpisource.status
} &

# start the "driver" CMSSW job(s)
{
  #mpirun --mca pmix_server_uri file:server.uri -n 1 -- ...
  cmsRun $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/testMPIDriver.py &> mpidriver.log
  echo $? > mpidriver.status
} &

# wait for all CMSSW jobs to finish
wait

# print the jobs' output and check the jobs' exit status
# pr -m -t -w 240 mpidriver.log mpisource.log
echo '============ testMPIDriver ============='
cat mpidriver.log
MPIDRIVER_STATUS=$(< mpidriver.status)
echo '========================================'
echo
echo '============ testMPISource ============='
cat mpisource.log
MPISOURCE_STATUS=$(< mpisource.status)
echo '========================================'

# stop the MPI server and cleanup the URI file
kill $SERVER_PID

popd > /dev/null
exit $((MPISOURCE_STATUS > MPIDRIVER_STATUS ? MPISOURCE_STATUS : MPIDRIVER_STATUS))
