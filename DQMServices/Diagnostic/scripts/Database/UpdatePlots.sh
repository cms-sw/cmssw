#!/bin/sh

source ${BaseDir}/MakeAllPlots.sh

function UpdatePlots ()
{
    local StorageDir=$4

    if [ ! -e ${StorageDir}/backup ]; then
	mkdir ${StorageDir}/backup
    fi

    argument1=`echo $1`
    argument2=`echo $2`

    local CMSSW_version=$3
    echo "args: $argument1 $argument2 $CMSSW_version"

    # Create new trends for the last 40 runs
    MakeAllPlots "${argument1}" "${argument2}" $CMSSW_version
    echo "MakeAllPlots ${argument1} ${argument2} $CMSSW_version"
    rm -rf ${StorageDir}/backup/Last40Runs
    mv ${StorageDir}/Last40Runs ${StorageDir}/backup
    mv HistoricDQMPlots ${StorageDir}/Last40Runs
    cp index.html ${StorageDir}/Last40Runs

    # Create new trends for all runs
    MakeAllPlots "${argument1}" "${argument2}" $CMSSW_version 1 20000000000
    rm -rf ${StorageDir}/backup/AllRuns
    mv ${StorageDir}/AllRuns ${StorageDir}/backup
    mv HistoricDQMPlots ${StorageDir}/AllRuns
    cp index.html ${StorageDir}/AllRuns
}
