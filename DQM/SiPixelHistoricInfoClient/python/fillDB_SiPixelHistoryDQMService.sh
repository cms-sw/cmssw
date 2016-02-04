#!/bin/sh

DirName=$1
rootFileList=(`ls  $1 | grep ".root" | sort`)


k=0
ListSize=${#rootFileList[*]}

while [ "$k" -lt "$ListSize" ]
  do
  rootFile=${rootFileList[$k]}
  runNumber=`echo ${rootFile} | awk -F "R00" '{print $2}' | awk -F"_" '{print int($1)}'`
  echo -e "\n\n\nprocessing " $rootFile " for runNr " ${runNumber} "\n\n"
     

  cat template_SiPixelHistoryDQMService_cfg.py | sed -e "s@RUNNUMBER@${runNumber}@g" -e "s@FILENAME@$DirName/$rootFile@" > RunMe_${runNumber}.py

  cmsRun RunMe_${runNumber}.py > Run_${runNumber}.log
  [ "$?" != "0" ] && echo -e "Problem found in the processing. please have a look at \nRun_${runNumber}.log" && exit
  rm RunMe_${runNumber}.py

  let "k+=1"
  done

echo "done."

