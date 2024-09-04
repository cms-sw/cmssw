#!/bin/bash

function die { echo $1: status $2 ; exit $2; }
python3 $CMSSW_BASE/src/CondCore/SiStripPlugins/scripts/G2GainsValidator.py  || die "Failure running G2GainsValidator.py" $?
