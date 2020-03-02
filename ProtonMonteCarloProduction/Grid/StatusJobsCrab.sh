#!/bin/bash

search_dir=`pwd` 
today="$( date +"%Y%m%d" )" 

printf "\n[CRAB STATUS]\n\tChecking status...\n\n"

source /cvmfs/cms.cern.ch/crab3/crab.sh
voms-proxy-init --voms cms

for entry in "$search_dir"/crab_*/*
do
  crab status -d $entry >> crab_status_$today.txt
done

printf "\n\tPlease, check the output crab_status_"$today".txt\n\n"

