#!/bin/sh

RunStart=$1
RunEnd=$2

Password=XXXPUTPASSWORDHEREXXX

BasePlotOutDir=`pwd`/./HistoricDQMPlots
mkdir -pv $BasePlotOutDir

BaseDir=./DQM
ListOfDets=(SiStripHistoricInfoClient
            TrackingHistoricInfoClient
            SiPixelHistoricInfoClient)

TagName=(HDQM_SiStrip_V2
         HDQM_Tracking_V1
         HDQM_SiPixel_V2)

k=0
ListSize=${#ListOfDets[*]}

while [ "$k" -lt "$ListSize" ]
do
  Det=${ListOfDets[$k]}
  echo Running on $Det

  ThisDir=`pwd`
  cd $BaseDir/$Det/scripts
  if [ $2 ]; then
    ./MakePlots.sh ${TagName[$k]} $Password $RunStart $RunEnd
  elif [ $1 ]; then
    ./MakePlots.sh ${TagName[$k]} $Password $RunStart
  else
    ./MakePlots.sh ${TagName[$k]} $Password 40
  fi
  mv -v CurrentPlots $BasePlotOutDir/Plots_$Det
  cd $ThisDir



  let "k+=1"
done


