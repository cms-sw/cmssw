#!/bin/bash
if [ $# -ne 3 ]
then
    echo "Three arguments required. Nothing done."
else
	# read files belonging to one run, print file: in front of them, get them on a single line, separated by commas
	filesToMerge=`eval 'ls /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/ | grep Run${1}_"[0-9][0-9]"*.root | sed -e "s/^/file:\/eos\/project\/c\/ctpps\/subsystems\/Pixel\/RPixTracking\/EfficiencyCalculation2018\/ReRecoOutputTmp_CMSSW_10_6_2\//" | awk "{print}" ORS=","' `
	# one terabyte file size limit
	cd ${3}/src
	eval `scramv1 runtime -sh`
	edmCopyPickMerge inputFiles=${filesToMerge::-1} outputFile=${2} maxSize=1000000000
	if [ $? -eq 0 ]
	then
		filesToDelete=`eval 'ls /eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/ | grep Run${1}_"[0-9][0-9]"*.root | sed -e "s/^/\/eos\/project\/c\/ctpps\/subsystems\/Pixel\/RPixTracking\/EfficiencyCalculation2018\/ReRecoOutputTmp_CMSSW_10_6_2\//"'`
		rm $filesToDelete
		exit 0
	else
		echo "An error during edmCopyPickMerge occurred!"
		exit 1
	fi
fi