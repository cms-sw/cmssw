#!/bin/bash

echo "create proxy: voms-proxy-init -voms cms"
voms-proxy-init -voms cms

echo "setup crab"
echo "crab 3 :: source /cvmfs/cms.cern.ch/crab3/crab.sh"
echo "crab 2 :: source /cvmfs/cms.cern.ch/crab/crab.sh"

# CRAB2 for slc6_amd64_gcc472 and lower ...
#################################################
echo "--> will use crab 2 for now"
source /cvmfs/cms.cern.ch/crab/crab.sh
crab --version

# CRAB3 only for slc6_amd64_gcc482 and higher ...
#################################################
# echo "--> will use crab 3"
# source /cvmfs/cms.cern.ch/crab3/crab.sh
# crab --version
