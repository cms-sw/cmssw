#########################################################################################
# This function performs the loop on all the files and uses the fillDB_HistoryDQMService
# function to execute the cmssw cfg.
#########################################################################################

source ${BaseDir}/fillDB_HistoryDQMService.sh
source ${BaseDir}/TakeLastVersion.sh

function SubDetScript ()
{
    local DetName=$1
    local TagName=$2
    local Database=$3
    local AuthenticationPath=$4
    local FirstRun=$5
    local LastRun=$6
    local CMSSW_version=$7
    local TemplatesDir=$8
    local SourceDir=$9


    # echo
    # echo "DetName=${DetName}"
    # echo "TagName=${TagName}"
    # echo "Database=${Database}"
    # echo "AuthenticationPath=${AuthenticationPath}"
    # echo "FirstRun=${FirstRun}"
    # echo "LastRun=${LastRun}"
    # echo "CMSSW_version=${CMSSW_version}"
    # echo
    # echo `pwd`

    if [ ! -d ${DetName} ]; then
	mkdir ${DetName}
    fi

    # For now we copy it back everytime, so that any change done in the main dir is automatically propagated in all subdirs
    cp ${TemplatesDir}/template_${DetName}HistoryDQMService_cfg.py ${DetName}
    cd ${DetName}

    local ProcessedFileList=HDQMProcessed_${TagName}.txt
    local TempProcessed=ToProcess_${DetName}.txt

    find -L ${SourceDir} -name "*.root" | awk -F"/" '{a=$0; sub($NF,"",a); print a}' > $TempProcessed
    # local ListOfDirs=(`cat $ProcessedFileList $TempProcessed | sort | uniq -u | awk -F"/" '{a=$0; sub($NF,"",a); print a}'`)
    # Declare this specifically as an array
    declare -a ListOfDirs
    # local ListOfDirs=($(cat $ProcessedFileList $TempProcessed | sort | uniq))
    local ListOfDirs=($(cat $TempProcessed | sort | uniq))

    local Update=0

    local k=0
    local ListSize=${#ListOfDirs[*]}

    echo "Number of runs to process: $ListSize"

    # while [ "$k" -lt 2 ]
    while [ "$k" -lt "$ListSize" ]
	do
	Dir=${ListOfDirs[$k]}

	# Internally runs that must not be processed are skipped. The return code tells if something has been processed.
	# Therefore, looking only at the ListSize is not sufficient to decide if to update the histograms.


	fillDB_HistoryDQMService ${DetName} ${TagName} ${Dir} ${Database} ${AuthenticationPath} ${FirstRun} ${LastRun} ${ProcessedFileList} ${CMSSW_version}


	local Update=$?

	let "k+=1"
	# echo "$k/$ListSize"
    done

    rm $TempProcessed

    echo "All done."

    # Important: revert to the initial dir
    cd -

    return ${Update}
}
