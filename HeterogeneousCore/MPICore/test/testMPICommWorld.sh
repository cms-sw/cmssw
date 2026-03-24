#! /bin/bash

# Shell script for testing CMSSW over MPI.

function usage() {
  echo "usage:"
  echo "  $(basename $0) config.py [config.py ...]"
  echo
}

# Make sure the CMSSW environment has been loaded.
if [ -z "$CMSSW_BASE" ]; then
  eval `scram runtime -sh`
fi

if [ $# == 0 ]; then
  usage
  echo "$(basename $0): error: please specify at least one argument."
  exit 1
fi

# Check that the arguments exist, and convert them to absolute paths.
CONFIGS=()
for CFG in "$@"; do
  if ! [ -f "$CFG" ]; then
    echo "$(basename $0): error: invalid argument '$CFG'"
    exit 1
  fi
  CONFIGS+=($(realpath "$CFG"))
done

# The CM pml leads to silent communication failures on some machines.
# Until this is understood and fixed, keep it disabled.
export OMPI_MCA_pml='^cm'

# Build the mpirun command line
CMD="-n 1 cmsRun ${CONFIGS[0]}" 
for CFG in "${CONFIGS[@]:1}"; do
    CMD+=" : -n 1 cmsRun ${CFG}"
done

# Launch the cmsRun processes
# The timeout is to prevent indefinite hangs if one process crashes without the
# other sides being aware of it.
timeout --signal=SIGTERM 120 mpirun $CMD
