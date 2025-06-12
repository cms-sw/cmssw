#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Short Track Validation..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testShortenedTrackValidation_cfg.py maxEvents=1000 || die "Failure running testShortenedTrackValidation_cfg.py" $?
