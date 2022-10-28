#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo " TESTING Primary Vertex Validation run-by-run submission ..."
submitPVValidationJobs.py -j UNIT_TEST -D /HLTPhysics/Run2016C-TkAlMinBias-07Dec2018-v1/ALCARECO -i ${LOCAL_TEST_DIR}/testPVValidation_Relvals_DATA.ini -r --unitTest || die "Failure running PV Validation run-by-run submission" $?

printf "\n\n"

echo " TESTING Split Vertex Validation submission ..."
submitPVResolutionJobs.py -j UNIT_TEST -D /JetHT/Run2018C-TkAlMinBias-12Nov2019_UL2018-v2/ALCARECO -i ${LOCAL_TEST_DIR}/PVResolutionExample.ini --unitTest || die "Failure running Split Vertex Validation submission" $?
