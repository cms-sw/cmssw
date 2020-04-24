#!/bin/bash



if [ ! $# -ge 1 ]; then
  echo "Usage: $0 iterationNumber"
  echo "Usage: $0 iterationNumber setBaseline"
  exit 1
fi

iterationNumber="$1"
setBaseline="False"
if [ $# == 2 ]; then
  setBaseline="$2";
  if [[ ! "$setBaseline" == False ]] && [[ ! "$setBaseline" == True ]] ; then
    echo "Invalid argument for setBaseline: $setBaseline"
    exit 2
  fi
fi

echo "Iteration number: $1"
echo "Set Baseline: ${setBaseline}"
echo




#######################################################
## Config for summary step
cmsRunOptions=" iterNumber=$iterationNumber setBaseline=$setBaseline"
echo "$cmsRunOptions"

summaryTemplateFile="${CMSSW_BASE}/src/Alignment/APEEstimation/test/cfgTemplate/summaryTemplate.bash"

summaryFile="${CMSSW_BASE}/src/Alignment/APEEstimation/test/batch/workingArea/summary.bash"
cat $summaryTemplateFile |sed "s/_THE_COMMANDS_/${cmsRunOptions}/g" > $summaryFile








#######################################################
## Create final output directory


ROOTFILEBASE="$CMSSW_BASE/src/Alignment/APEEstimation/hists"

if [[ "$setBaseline" == True ]] ; then
  fileDir="${ROOTFILEBASE}/Design/baseline"
  
  # If there is already output from previous studies, move it
  if [ -d "${fileDir}" ] ; then
    mv ${fileDir} ${ROOTFILEBASE}/Design/baseline_old ;
  fi
  mkdir ${ROOTFILEBASE}/Design
  mkdir ${fileDir}
else
  fileDir="${ROOTFILEBASE}/workingArea/iter${iterationNumber}"
  
  # If there is already output from previous studies, move it
  if [ -d "${fileDir}" ] ; then
    mv ${fileDir} ${ROOTFILEBASE}/workingArea/iter${iterationNumber}_old
  fi
  if [ -a /afs/cern.ch/user/h/hauk/scratch0/apeStudies/apeObjects/apeIter${iterationNumber}.db ] ; then
    mv /afs/cern.ch/user/h/hauk/scratch0/apeStudies/apeObjects/apeIter${iterationNumber}.db /afs/cern.ch/user/h/hauk/scratch0/apeStudies/apeObjects/apeIter${iterationNumber}_old.db
  fi
  mkdir ${fileDir}
  
  if [ "$iterationNumber" -ne 0 ] ; then
    declare -i nIterDecrement=${iterationNumber}-1
    cp ${ROOTFILEBASE}/workingArea/iter${nIterDecrement}/allData_iterationApe.root ${fileDir}/.
  fi
fi






#######################################################
## Add root files from step1 and delete them, keep only summed file


hadd ${fileDir}/allData.root ${ROOTFILEBASE}/workingArea/*.root
if [ $? -eq 0 ] ; then
  rm ${ROOTFILEBASE}/workingArea/*.root
fi




