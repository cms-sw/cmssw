#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

for file in ${CMSSW_BASE}/src/FWCore/ParameterSet/python/*.py
do
  bn=`basename $file`
  if [ "$bn" != "__init__.py" ]; then
     bnm=${bn%.*} 
     python -m FWCore.ParameterSet."$bnm" || die "unit tests for $bn failed" $?
  fi
done
