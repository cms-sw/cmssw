#!/bin/sh

DetName=$1
TagName=$2
Database=$3
AuthenticationPath=$4
FirstRun=$5
LastRun=$6

SourceDir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/PromptReco/

if [ ! -d ${DetName} ]; then
    mkdir ${DetName}
    # if [ ! -e ${DetName}/fillDB_HistoryDQMService.sh ]; then
    #    cp fillDB_HistoryDQMService.sh ${DetName}
    # fi
fi

# For now we copy it back everytime, so that any change done in the main dir is automatically propagated in all subdirs
cp fillDB_HistoryDQMService.sh ${DetName}
# The same is true for the template cfg.
cp Templates/template_${DetName}HistoryDQMService_cfg.py ${DetName}
cd ${DetName}


ProcessedFileList=HDQMProcessed_${TagName}.txt
TempProcessed=ToProcess_${DetName}.txt

find ${SourceDir} -name "*.root" | awk -F"/" '{a=$0; sub($NF,"",a); print a}' > $TempProcessed
ListOfDirs=(`cat $ProcessedFileList $TempProcessed | sort | uniq -u | awk -F"/" '{a=$0; sub($NF,"",a); print a}'`)


k=0
ListSize=${#ListOfDirs[*]}

echo "Number of runs to process: $ListSize"

# while [ "$k" -lt 2 ]
while [ "$k" -lt "$ListSize" ]
  do
  Dir=${ListOfDirs[$k]}
  # echo Running on $Dir

  ./fillDB_HistoryDQMService.sh ${DetName} ${TagName} ${Dir} ${Database} ${AuthenticationPath} ${FirstRun} ${LastRun} ${ProcessedFileList}

  # echo ${Dir} >> $ProcessedFileList

  let "k+=1"
done

rm $TempProcessed

echo "All done."


