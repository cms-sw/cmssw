#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

## copy into local sqlite file the ideal alignment
echo "COPYING locally Ideal Alignment ..."
conddb --yes --db pro copy TrackerAlignment_Upgrade2017_design_v4 --destdb myfile.db
conddb --yes --db pro copy TrackerAlignmentErrorsExtended_Upgrade2017_design_v0 --destdb myfile.db

echo " TESTING Primary Vertex Validation run-by-run submission ..."
submitPVValidationJobs.py -j UNIT_TEST -D /HLTPhysics/Run2016C-TkAlMinBias-07Dec2018-v1/ALCARECO -i ${LOCAL_TEST_DIR}/testPVValidation_Relvals_DATA.ini -r --unitTest || die "Failure running PV Validation run-by-run submission" $?

