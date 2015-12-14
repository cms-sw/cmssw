#!/bin/sh

eos="/afs/cern.ch/project/eos/installation/cms/bin/eos.select"

if [ ! $# == 1 ]; then
  echo "Usage: $0 sample"
  exit 1
fi


sample="$1"

directory=""
if [[ "$sample" == data1 ]] ; then directory="/store/caf/user/cschomak/data/SingleMu/Run2012A-22Jan2013_v5/"
elif [[ "$sample" == data2 ]] ; then directory="/store/caf/user/cschomak/data/SingleMu/Run2012B-22Jan2013_v5/"
elif [[ "$sample" == data3 ]] ; then directory="/store/caf/user/cschomak/data/SingleMu/Run2012C-22Jan2013_v5/"
elif [[ "$sample" == data4 ]] ; then directory="/store/caf/user/cschomak/data/SingleMu/Run2012D-22Jan2013_v5/"
elif [[ "$sample" == qcd ]] ; then directory="/store/caf/user/cschomak/mc/Summer12_v5/qcd/"
elif [[ "$sample" == wlnu ]] ; then directory="/store/caf/user/cschomak/mc/test/wlnu/"
elif [[ "$sample" == zmumu10 ]] ; then directory="/store/caf/user/cschomak/mc/Summer12_v5/zmumu10/"
elif [[ "$sample" == zmumu20 ]] ; then directory="/store/caf/user/cschomak/mc/Summer12_v5/zmumu20/"
elif [[ "$sample" == zmumu50 ]] ; then directory="/store/caf/user/cschomak/"
else
  echo "Invalid dataset: $sample"
  exit 2
fi

filebase="${directory}apeSkim"

filesuffix=".root"

tempFile="/tmp/cschomak/temp.root"
if [ -f $tempFile ] ; then mv $tempFile ${tempFile}.old; fi

## increment counter
declare -i counter=1

inputFilename="${filebase}${filesuffix}"
outputFilename="${filebase}${counter}${filesuffix}"

xrdcp root://eoscms//eos/cms${inputFilename} $tempFile

if [ ! -f $tempFile ] ; then echo "Last file reached: 0"; exit 0; fi
xrdcp $tempFile root://eoscms//eos/cms${outputFilename}
if [ $? -eq 0 ] ; then
  $eos rm ${filebase}${filesuffix}
fi
rm $tempFile


while [ $counter -le 9 ]
do
  declare -i counterIncrement=${counter}+1
  
  inputFilename="${filebase}00${counter}${filesuffix}"
  outputFilename="${filebase}${counterIncrement}${filesuffix}"
  
  xrdcp root://eoscms//eos/cms${inputFilename} $tempFile
  if [ ! -f $tempFile ] ; then echo "Last file reached: ${counter}"; exit 0; fi
  xrdcp $tempFile root://eoscms//eos/cms${outputFilename}
  if [ $? -eq 0 ] ; then
    $eos rm $inputFilename
  fi
  rm $tempFile
  
  counter=$counter+1
done


## increment counter after first 10 files
declare -i counterTen=10

while [ $counterTen -le 99 ]
do
  declare -i counterTenIncrement=${counterTen}+1
  
  inputFilename="${filebase}0${counterTen}${filesuffix}"
  outputFilename="${filebase}${counterTenIncrement}${filesuffix}"
  
  xrdcp root://eoscms//eos/cms${inputFilename} $tempFile
  if [ ! -f $tempFile ] ; then echo "Last file reached: ${counterTen}"; exit 0; fi
  xrdcp $tempFile root://eoscms//eos/cms${outputFilename}
  if [ $? -eq 0 ] ; then
    $eos rm $inputFilename
  fi
  rm $tempFile
  
  counterTen=$counterTen+1
done






