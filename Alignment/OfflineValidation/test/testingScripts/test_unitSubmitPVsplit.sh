#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo " TESTING Split Vertex Validation submission ..."
submitPVResolutionJobs.py -j UNIT_TEST -D /JetHT/Run2022B-TkAlJetHT-PromptReco-v1/ALCARECO \
  -i ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/PVResolutionExample.ini --unitTest || die "Failure running Split Vertex Validation submission" $?

echo -e "\n\n TESTING Primary Vertex Split script execution ..."
# Define script name
scriptName="batchHarvester_Prompt_0.sh"

# Create directory if it doesn't exist
mkdir -p "./testExecution"

# Check if the script exists and is a regular file
if [ -f "./BASH/${scriptName}" ]; then
    # Copy script to the test execution directory
    cp -pr "./BASH/${scriptName}" "./testExecution/"
else
    # Emit a warning if the script doesn't exist or is not a regular file
    echo "Warning: Script '${scriptName}' not found or is not a regular file. Skipping excution of further tests."
    exit 0
fi

# Change directory to the test execution directory
cd "./testExecution" || exit 1

# Execute the script and handle errors
./"${scriptName}" || die "Failure running PVSplit script" $?

# Dump to screen the content of the log file
cat log*.out
