#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

for file in ${CMSSW_BASE}/src/FWCore/ParameterSet/python/*.py
do
  python "$file" || die 'unit tests for $file failed' $?
done
