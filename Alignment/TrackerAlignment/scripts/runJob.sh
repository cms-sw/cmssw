#!/bin/bash

if [ "$1" == "-h" ] || [ "$1" == "--help" ]
then
  printf "\nUsage: "
  printf "runJob.sh [Queue(optional)]\n\n"
  exit 1;
fi

QUEUE=$1

if [ $# -lt 1 ] 
then
printf "\nJob is submitted to default Queue '8nh'.\n"
printf "If you want to submit job in specific queue then please use :\n\n"
printf "./run.sh [Queue name]\n\n"
bsub -q 8nh < $CMSSW_BASE/src/Alignment/TrackerAlignment/scripts/jobConfig.sh 

else 
bsub -q $QUEUE < $CMSSW_BASE/src/Alignment/TrackerAlignment/scripts/jobConfig.sh

fi

