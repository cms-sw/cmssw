#!/bin/sh

function MakeAllPlots ()
{
    declare -a SubDets
    local SubDets=($(echo "$2"))
    declare -a TagNames
    local TagNames=($(echo "$1"))

    local CMSSW_version=$3

    echo "SubDets = $1"
    echo "TagNames = $2"

    # source /afs/cern.ch/cms/sw/cmsset_default.sh

    local RunStart=$4
    local RunEnd=$5

    local Password=PASSWORD

    local BasePlotOutDir=`pwd`/./HistoricDQMPlots
    mkdir -pv $BasePlotOutDir

    local LocalBaseDir=${CMSSW_version}/src/DQMServices/Diagnostic

    local k=0
    local ListSize=${#SubDets[*]}
    echo "ListSize = $ListSize"

    while [ "$k" -lt "$ListSize" ]
    do
	local Det="${SubDets[$k]}HistoricInfoClient"
	local Macro="${SubDets[$k]}HDQMInspector"
	echo Running on $Det

	echo "RUNSTART = $RunStart"
	echo "RUNEND = $RunEnd"

	local ThisDir=`pwd`
	cd ${LocalBaseDir}/scripts
	if [ ${RunEnd} ]; then
	    ./MakePlots.sh ${Macro} ${Database} ${TagNames[$k]} $Password $RunStart $RunEnd
	elif [ ${RunStart} ]; then
	    ./MakePlots.sh ${Macro} ${Database} ${TagNames[$k]} $Password $RunStart
	else
	    ./MakePlots.sh ${Macro} ${Database} ${TagNames[$k]} $Password 40
	fi
	mv -v CurrentPlots $BasePlotOutDir/Plots_$Det
	cd $ThisDir

	let "k+=1"
    done
}
