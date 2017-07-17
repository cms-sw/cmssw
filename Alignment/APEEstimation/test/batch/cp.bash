#!/bin/bash

if [ ! $# == 1 ]; then
  echo "Usage: $0 sample"
  exit 1
fi

sample="$1"

directory=""

if [[ "$sample" == data1 ]] ; then directory="/store/caf/user/ajkumar/data/SingleMu/Run2012A-22Jan2013/"
elif [[ "$sample" == data2 ]] ; then directory="/store/caf/user/ajkumar/data/SingleMu/Run2012B-22Jan2013/"
elif [[ "$sample" == data3 ]] ; then directory="/store/caf/user/ajkumar/data/SingleMu/Run2012C-22Jan2013/"
elif [[ "$sample" == data4 ]] ; then directory="/store/caf/user/ajkumar/data/SingleMu/Run2012D-22Jan2013/"
elif [[ "$sample" == qcd ]] ; then directory="/store/caf/user/ajkumar/mc/Summer12_v1/qcd/"
elif [[ "$sample" == wlnu ]] ; then directory="/store/caf/user/ajkumar/mc/Summer12_v1/wlnu/"
elif [[ "$sample" == zmumu10 ]] ; then directory="/store/caf/user/ajkumar/mc/Summer12_v1/zmumu10/"
elif [[ "$sample" == zmumu20 ]] ; then directory="/store/caf/user/ajkumar/mc/Summer12_v1/zmumu20/"
else
  echo "Invalid dataset: $sample"
  exit 2
fi


ls -l

cd $CMSSW_BASE/src
if [[ "$SHELL" == /bin/sh || "$SHELL" == /bin/bash || "$SHELL" == /bin/zsh ]] ; then
  eval `scram runtime -sh`
elif [[ "$SHELL" == /bin/csh || "$SHELL" == /bin/tcsh ]] ; then
  eval `scram runtime -csh`
else
  echo "Unknown shell: $SHELL"
  echo "cannot set CMSSW environment, stop processing"
  exit 5
fi
cd -

#cmsRun $CMSSW_BASE/src/ApeEstimator/ApeEstimator/test/SkimProducer/skimProducer_cfg.py isTest=False sample=$sample
#cmsRun $CMSSW_BASE/src/ApeEstimator/ApeEstimator/test/SkimProducer/skimProducer_cfg.py isTest=False useTrackList=False sample=$sample

ls -l


for file in *.root;
do
  xrdcp $file root://eoscms//eos/cms${directory}${file}
done








