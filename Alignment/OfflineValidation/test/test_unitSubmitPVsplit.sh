#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo " TESTING Split Vertex Validation submission ..."
submitPVResolutionJobs.py -j UNIT_TEST -D /JetHT/Run2018C-TkAlMinBias-12Nov2019_UL2018-v2/ALCARECO -i ${LOCAL_TEST_DIR}/PVResolutionExample.ini --unitTest || die "Failure running Split Vertex Validation submission" $?
