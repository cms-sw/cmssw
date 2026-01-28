#! /bin/bash

# Shell script for testing CMSSW over MPI.
CONTROLLER=$(realpath $1)
FOLLOWER=$(realpath $2)

# Make sure the CMSSW environment has been loaded.
if [ -z "$CMSSW_BASE" ]; then
  eval `scram runtime -sh`
fi

# The CM pml leads to silent communication failures on some machines.
# Until this is understood and fixed, keep it disabled.
export OMPI_MCA_pml='^cm'

# Launch the controller and follower processes
mpirun -n 1 cmsRun ${CONTROLLER} : -n 1 cmsRun ${FOLLOWER}
