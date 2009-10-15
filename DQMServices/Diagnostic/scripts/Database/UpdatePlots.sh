#!/bin/sh

source ${BaseDir}/MakeAllPlots.sh

function UpdatePlots ()
{
    local StorageDir=/storage/data2/SiStrip/historic_dqm/HistoryDQM

    if [ ! -e ${StorageDir}/backup ]; then
	mkdir ${StorageDir}/backup
    fi

    argument1=`echo $1`
    argument2=`echo $2`

    local CMSSW_version=$3

    # Create new trends for the last 40 runs
    MakeAllPlots "${argument1}" "${argument2}" $CMSSW_version
    rm -rf ${StorageDir}/backup/Last40Runs
    mv ${StorageDir}/Last40Runs ${StorageDir}/backup
    mv HistoricDQMPlots ${StorageDir}/Last40Runs
    cp index.html ${StorageDir}/Last40Runs

    # Done by hand for the special case of RPC
    cp RPC/index.html ${StorageDir}/Last40Runs/Plots_RPCHistoricInfoClient

    # Create new trends for all runs
    MakeAllPlots "${argument1}" "${argument2}" $CMSSW_version 1 20000000000
    rm -rf ${StorageDir}/backup/AllRuns
    mv ${StorageDir}/AllRuns ${StorageDir}/backup
    mv HistoricDQMPlots ${StorageDir}/AllRuns
    cp index.html ${StorageDir}/AllRuns

    # Done by hand for the special case of RPC
    cp RPC/index.html ${StorageDir}/AllRuns/Plots_RPCHistoricInfoClient
}
