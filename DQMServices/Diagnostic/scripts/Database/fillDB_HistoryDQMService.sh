#!/bin/sh

DetName=$1
TagName=$2
DirName=$3
Database=$4
AuthenticationPath=$5
FirstRun=$6
LastRun=$7
ProcessedFileList=$8

# The "$" sign tells it to match that expression at the end of a line
rootFileList=(`ls ${DirName} | grep ".root$" | sort`)

k=0
ListSize=${#rootFileList[*]}

while [ "$k" -lt "$ListSize" ]
  do
  rootFile=${rootFileList[$k]}
  runNumber=`echo ${rootFile} | awk -F "R00" '{print $2}' | awk -F"_" '{print int($1)}'`
  lessThenLastRun=0
  if [ ${LastRun} -eq -1 ] || [ ${runNumber} -le ${lastRun} ]; then
    lessThenLastRun=1
  fi
  if [ ${runNumber} -ge ${FirstRun} ] && [ ${lessThenLastRun} -eq 1 ]; then
    echo -e "\n\n\nprocessing " $rootFile " for runNr " ${runNumber} "\n\n"
    # echo "processing $rootFile for runNr ${runNumber}"

    cat template_${DetName}HistoryDQMService_cfg.py | sed -e "s@RUNNUMBER@${runNumber}@g" -e "s@FILENAME@$DirName/$rootFile@" -e "s@TAGNAME@${TagName}@g" -e "s@DATABASE@${Database}@" -e "s@AUTHENTICATIONPATH@${AuthenticationPath}@" > Run_${DetName}_${runNumber}.py

    cmsRun Run_${DetName}_${runNumber}.py > Run_${DetName}_${runNumber}.log
    # if [ "$?" != "0" ]; then
    # if [ $? -ne 0 ]; then
      # echo -e "Problem found in the processing (exit code = $?). please have a look at \nRun_${DetName}_${runNumber}.log" && exit
    # else
      # The file is processed only if the job was successful
      echo ${DirName} >> ${ProcessedFileList}
      echo "done."
    # fi
  else
    echo "Skipping run $runNumber"
  fi
  #   # rm Run_${DetName}_${runNumber}.py

  let "k+=1"
  # fi
done

