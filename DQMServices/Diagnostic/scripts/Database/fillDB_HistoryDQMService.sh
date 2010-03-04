###############################################################################
# This function performs the actual writing to the database of the quantities.
# It does a check for each run if it is in the specified range.
# The check is done internally because the extraction of the run number
# from the directory and file name is also done internally.
###############################################################################

function fillDB_HistoryDQMService ()
{
    local DetName=$1
    local TagName=$2
    local DirName=$3
    local Database=$4
    local AuthenticationPath=$5
    local FirstRun=$6
    local LastRun=$7
    local ProcessedFileList=$8
    local CMSSW_version=$9

    echo "DirName: $DirName"

    # echo
    # echo "PRINTING INFO IN FILLDB"
    # echo "DetName=$DetName"
    # echo "TagName=$TagName"
    # echo "DirName=$DirName"
    # echo "Database=$Database"
    # echo "AuthenticationPath=$AuthenticationPath"
    # echo "FirstRun=$FirstRun"
    # echo "LastRun=$LastRun"
    # echo "ProcessedFileList=$ProcessedFileList"
    # echo "CMSSW_version=$CMSSW_version"
    # echo
    # echo


    local Update=0

    # The "$" sign tells it to match that expression at the end of a line
    declare -a rootFileList



    local rootFileListName="TempList.txt"
    TakeLastVersion ${DirName} ${rootFileListName}

#rootFileList=(`cat ${rootFileListName} | grep ".root$" | sort`)
#echo "rootFileList: $rootFileList"

    declare -a UniqFileList=(`cat $rootFileListName $ProcessedFileList | grep .root | sort | uniq -u`)
#echo `echo $rootFileList | sed -e "s-.root-.root\n-g" | cat $ProcessedFileList - | sort | uniq -u`


    local k=0
    local ListSize=${#UniqFileList[*]}
    echo "Processing $ListSize runs"

    while [ "$k" -lt "$ListSize" ]
        do
	local rootFile=${UniqFileList[$k]}
	local runNumber=`echo ${rootFile} | awk -F "R00" '{print $2}' | awk -F"_" '{print int($1)}'`
	local lessThenLastRun=0
	if [ ${LastRun} -eq -1 ] || [ ${runNumber} -le ${LastRun} ]; then
	    local lessThenLastRun=1
	fi
	if [ ${runNumber} -ge ${FirstRun} ] && [ ${lessThenLastRun} -eq 1 ]; then
	    echo -e "\n\n\nprocessing " $rootFile " for runNr " ${runNumber} "\n\n"
	    # echo "processing $rootFile for runNr ${runNumber}"
	    cat template_${DetName}HistoryDQMService_cfg.py | sed -e "s@RUNNUMBER@${runNumber}@g" -e "s@FILENAME@$rootFile@" -e "s@TAGNAME@${TagName}@g" -e "s@DATABASE@${Database}@" -e "s@AUTHENTICATIONPATH@${AuthenticationPath}@" > Run_${DetName}_${runNumber}.py
      cmsRun Run_${DetName}_${runNumber}.py > Run_${DetName}_${runNumber}.log
	    # if [ "$?" != "0" ]; then
	    # if [ $? -ne 0 ]; then
		# echo -e "Problem found in the processing (exit code = $?). please have a look at \nRun_${DetName}_${runNumber}.log" && exit
	    # else
	    # The file is processed only if the job was successful
	    echo ${rootFile} >> ${ProcessedFileList}
	    echo "done."
	    # fi
	    local Update=1
	# else
	    # echo "Skipping run $runNumber"
	fi
	#   # rm Run_${DetName}_${runNumber}.py
	let "k+=1"
	# fi
    done

    return ${Update}
}
