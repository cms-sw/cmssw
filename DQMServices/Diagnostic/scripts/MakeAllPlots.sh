#!/bin/sh

source /afs/cern.ch/cms/sw/cmsset_default.sh

RunStart=$1
RunEnd=$2

#Password=<<!!INSERT_DATABASE_PASSWORD_HERE!!>>
Password=YHIX4CVVAJVRUXE6

BasePlotOutDir=`pwd`/./HistoricDQMPlots
mkdir -pv $BasePlotOutDir

BaseDir=$CMSSW_BASE/src/DQMServices/Diagnostic

ListOfExes=(SiStripHDQMInspector
            TrackingHDQMInspector
            SiPixelHDQMInspector
            )
ListOfDets=(SiStripHistoricInfoClient
            TrackingHistoricInfoClient
            SiPixelHistoricInfoClient
            )

TagName=(HDQM_SiStrip_V3
         HDQM_Tracking_V1
         HDQM_SiPixel_V3
         )

k=0
ListSize=${#ListOfDets[*]}

while [ "$k" -lt "$ListSize" ]
do
  Det=${ListOfDets[$k]}
  echo Running on $Det

  ThisDir=`pwd`
  cd $BaseDir/scripts
  if [ $2 ]; then
    ./MakePlots.sh ${ListOfExes[$k]} ${TagName[$k]} $Password $RunStart $RunEnd
  elif [ $1 ]; then
    ./MakePlots.sh ${ListOfExes[$k]} ${TagName[$k]} $Password $RunStart
  else
    ./MakePlots.sh ${ListOfExes[$k]} ${TagName[$k]} $Password 40
  fi
  mv -v CurrentPlots $BasePlotOutDir/Plots_$Det
  cd $ThisDir

  let "k+=1"
done
