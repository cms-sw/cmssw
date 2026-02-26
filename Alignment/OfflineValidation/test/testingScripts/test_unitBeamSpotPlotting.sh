#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo -e "\n\nTESTING BeamSpot Plotting Script"

python3 ${CMSSW_BASE}/src/Alignment/OfflineValidation/scripts/plotBeamSpotFromOfflineTag.py --unitTest || die "Failure running BeamSpot Plotting Script" $?
