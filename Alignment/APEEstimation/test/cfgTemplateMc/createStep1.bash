#!/bin/bash



if [ ! $# -ge 1 ]; then
  echo "Usage: $0 iterationNumber"
  echo "Usage: $0 iterationNumber lastIteration"
  exit 1
fi

export iterationNumber="$1"
export lastIteration="False"
if [ $# == 2 ]; then
  lastIteration="$2";
  if [[ ! "$lastIteration" == False ]] && [[ ! "$lastIteration" == True ]] ; then
    echo "Invalid argument for lastIteration: $lastIteration"
    exit 2
  fi
fi

echo "Iteration number: $1"
echo "LastIteration: ${lastIteration}"
echo





## Alignment
export alignmentRcd="misalTest"
#export alignmentRcd="idealAligned"
echo "Alignment Record: $alignmentRcd"
echo



## Script to create submit scripts for specific dataset
createStep1="${CMSSW_BASE}/src/Alignment/APEEstimation/test/cfgTemplate/writeSubmitScript.sh"

## identification name of dataset
export datasetName
## number of input files
export nFiles
## Input file base
cafDir="\/store\/caf\/user\/cschomak\/wlnu"
export inputBase


datasetName="wlnu"
inputBase="${cafDir}\/${datasetName}"
nFiles=9
bash $createStep1 $datasetName $nFiles $iterationNumber $lastIteration $alignmentRcd $inputBase
