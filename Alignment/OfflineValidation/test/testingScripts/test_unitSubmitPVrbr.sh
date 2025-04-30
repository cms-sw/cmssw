#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

## copy into local sqlite file the ideal alignment
echo "COPYING locally Ideal Alignment ..."
conddb --yes --db pro copy TrackerAlignment_Upgrade2017_design_v4 --destdb myfile.db
conddb --yes --db pro copy TrackerAlignmentErrorsExtended_Upgrade2017_design_v0 --destdb myfile.db

echo " TESTING Primary Vertex Validation run-by-run submission ..."
submitPVValidationJobs.py -j UNIT_TEST -D /HLTPhysics/Run2023D-TkAlMinBias-PromptReco-v2/ALCARECO \
  -i ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testPVValidation_Relvals_DATA.ini -r --unitTest || die "Failure running PV Validation run-by-run submission" $?

echo -e "\n\n TESTING Primary Vertex Validation script execution ..."
# Define script name
scriptName="PVValidation_testingOfflineGT_HLTPhysics_Run2023D_0.sh"

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
./"${scriptName}" || die "Failure running PVValidation script" $?
-- dummy change --
