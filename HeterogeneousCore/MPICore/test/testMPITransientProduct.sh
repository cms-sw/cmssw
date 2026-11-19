#! /bin/bash

# Products without a TrivialSerialisation plugin cannot be trivially serialised.
# If "persistent=false" is also set for these products in the corresponding
# classes_def.xml, these products cannot be serialised through ROOT either.
# Products that satisfy these two conditions have no serialisation mechanism,
# and thus cannot be transferred by the MPI modules.
#
# This test attempts to transfer a MPIToken, which satisfies these two
# conditions. The expected output is an exception thrown by the
# MPISenderPortable. This script looks for the exception message in the cmsRun
# output. If found, the test passes. Otherwise the test fails.


# Make sure the CMSSW environment has been loaded.
if [ -z "$CMSSW_BASE" ]; then
  eval `scram runtime -sh`
fi

OUTPUT=$("$CMSSW_BASE"/src/HeterogeneousCore/MPICore/test/testMPICommWorld.sh "$1" "$2" 2>&1)
STATUS=$?

echo "$OUTPUT"

if [ $STATUS -eq 0 ]; then
  echo "$(basename $0): error: cmsRun did not fail when it was expected to."
  exit 1
fi

if ! echo "$OUTPUT" | grep -q 'is transient (persistent = "false")'; then
  echo "$(basename $0): error: cmsRun failed for an unexpected reason."
  exit 1
fi

exit 0
