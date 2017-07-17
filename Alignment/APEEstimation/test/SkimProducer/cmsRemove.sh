#!/bin/sh

eos="/afs/cern.ch/project/eos/installation/cms/bin/eos.select"

#directory="/store/caf/user/hauk/data/mu/Run2010B_Dec22ReReco/"
directory="/store/caf/user/ajkumar/ApeSkim/zmumu50/"
#directory="/store/caf/user/hauk/data/Mu/Run2010A_Dec22ReReco/"
#directory="/store/caf/user/hauk/data/Mu/Run2010B_Dec22ReReco/"
#directory="/store/caf/user/hauk/mc/Qcd/"
#directory="/store/caf/user/hauk/mc/Wmunu/"
#directory="/store/caf/user/hauk/mc/Zmumu/"
#directory="/store/caf/user/hauk/mc/qcd/"
#directory="/store/caf/user/hauk/mc/wlnu/"
#directory="/store/caf/user/hauk/mc/zmumu/"
#directory="/store/caf/user/hauk/mc/ztautau/"

#directory="/store/caf/user/hauk/mc/ParticleGunMuon/RAW/"
#directory="/store/caf/user/hauk/mc/ParticleGunMuon/RECO/"
#directory="/store/caf/user/hauk/mc/ParticleGunAntiMuon/RAW/"
#directory="/store/caf/user/hauk/mc/ParticleGunAntiMuon/RECO/"





filebase="${directory}apeSkim"
#filebase="${directory}raw"
#sfilebase="${directory}reco"



filesuffix=".root"


## increment counter
declare -i counter=1


while [ $counter -le 1000 ]
do
  fullname="${filebase}${counter}${filesuffix}"
  
  $eos ls ${fullname}
  if [ $? -eq 0 ] ; then
    echo "Delete file: ${counter}";
    $eos rm ${fullname}
  else
    echo "Last file reached: ${counter}"; exit 0;
  fi
  
  counter=$counter+1
  
done






