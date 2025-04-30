#!/bin/bash
function die { echo $1: status $2; exit $2; }

# Check if doPede argument is provided and is true
if [[ "$1" == "--doPede" ]]; then
    doPede=true
else
    doPede=false
fi

echo -e "testing mille step ..."
(cmsRun ${SCRAM_TEST_PATH}/test_mille.py algoMode="mille") || die 'failed running test_mille.py (LAPACK)' $?

if [[ "$doPede" == "true" ]]; then
    echo -e "\n\ntesting pede step with sparseMINRES ..."
    (cmsRun ${SCRAM_TEST_PATH}/test_mille.py algoMode="pede" useLapack=False) || die 'failed running test_mille.py (MINRES)' $?

    echo -e "\n\ntesting pedes step with LAPACK ..."
    (cmsRun ${SCRAM_TEST_PATH}/test_mille.py algoMode="pede" useLapack=True) || die 'failed running test_mille.py (LAPACK)' $?
fi
-- dummy change --
