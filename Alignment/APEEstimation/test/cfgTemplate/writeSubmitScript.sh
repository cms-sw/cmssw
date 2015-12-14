#!/bin/tcsh



isData=false
isMc=false
isParticleGun=false
if [[ "$datasetName" == data1 ]] ||  [[ "$datasetName" == data2 ]]; then
  isData=true;
elif [[ "$datasetName" == qcd ]] || [[ "$datasetName" == wlnu ]] || [[ "$datasetName" == zmumu ]] || [[ "$datasetName" == ztautau ]]  || [[ "$datasetName" == zmumu10 ]] || [[ "$datasetName" == zmumu20 ]] || [[ "$datasetName" == zmumu50 ]]; then
  isMc=true;
elif [[ "$datasetName" == ParticleGunMuon ]] || [[ "$datasetName" == ParticleGunPion ]] || [[ "$datasetName" == ParticleGunAntiMuon ]] || [[ "$datasetName" == ParticleGunAntiPion ]] ; then
  isParticleGun=true;
else
  echo "Invalid dataset: $datasetName"
  exit 3
fi



## Batch submit file template
BATCH_TEMPLATEFILE="${CMSSW_BASE}/src/Alignment/APEEstimation/test/cfgTemplate/batchSubmitTemplate.tcsh"
BATCH_OUTPUTBASE1="${CMSSW_BASE}/src/Alignment/APEEstimation/test/batch/workingArea/${datasetName}BatchSubmit"
BATCH_OUTPUTSUFFIX=".tcsh"

helpFile1="help1.txt"
cat $BATCH_TEMPLATEFILE |sed "s/_THE_INPUTBASE_/root:\/\/eoscms\/\/eos\/cms\/${inputBase}/g" > $helpFile1




## increment counter
declare -i counter1=1

## number of files to create (maximum value of counter!!!)
while [ $counter1 -le ${nFiles} ]
do
  cmsRunOptions=" sample=$datasetName fileNumber=$counter1 iterNumber=$iterationNumber lastIter=$lastIteration alignRcd=$alignmentRcd"
  #echo "$cmsRunOptions"
  
  helpFile2="help2.txt"
  cat $helpFile1 |sed "s/_THE_COMMANDS_/${cmsRunOptions}/g" > $helpFile2
  
  theBatchFilename="${BATCH_OUTPUTBASE1}${counter1}${BATCH_OUTPUTSUFFIX}"
  cat $helpFile2 |sed "s/_THE_NUMBER_/${counter1}/g" > $theBatchFilename
  
  counter1=$counter1+1
done


echo "Sample, number of files: $datasetName, $nFiles"


rm $helpFile1
rm $helpFile2





