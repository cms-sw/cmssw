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
#export alignmentRcd="globalTag"
#~ export alignmentRcd="mp1791"
export alignmentRcd="hp1370"

echo "Alignment Record: $alignmentRcd"
echo



## Script to create submit scripts for specific dataset
createStep1="${CMSSW_BASE}/src/Alignment/APEEstimation/test/cfgTemplate/writeSubmitScript.sh"

## identification name of dataset
export datasetName
## number of input files
export nFiles
## Input file base
cafDir="\/store\/caf\/user\/cschomak\/SingleMu2015RunB"
cafDir2="\/store\/caf\/user\/cschomak\/DoubleMu2015RunB"
export inputBase


datasetName="data1"
inputBase="${cafDir}\/DataSingleMuonRun2015BPromptReco"
nFiles=9
bash $createStep1 $datasetName $nFiles $iterationNumber $lastIteration $alignmentRcd $inputBase


datasetName="data2"
inputBase="${cafDir2}\/DataDoubleMuonRun2015BPromptReco"
nFiles=1
bash $createStep1 $datasetName $nFiles $iterationNumber $lastIteration $alignmentRcd $inputBase



