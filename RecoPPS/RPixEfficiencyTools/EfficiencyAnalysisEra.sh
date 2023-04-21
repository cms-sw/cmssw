#!/bin/bash
if [ $# -ne 2 ]
then
    echo "Two arguments required. Nothing done."
else
	export CMSSW_BASE=${2}
	export SCRAM_ARCH=slc7_amd64_gcc700
	export X509_USER_PROXY=$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/x509up_u$UID
	addJSON_A="useJsonFile=True jsonFileName=/eos/project-c/ctpps/Operations/DataExternalConditions/2018/CMSgolden_2RPGood_anyarms_EraA.json"
	if true 
	then	
		addJSON="useJsonFile=True jsonFileName=/eos/project-c/ctpps/Operations/DataExternalConditions/2018/CMSgolden_2RPGood_anyarms_Era${1}.json"
		# addJSON="maxTracksInTagPot=1"
		fileSuffix="_reMiniAOD"
		outputFileSuffix=""
	fi

	cd $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/
	eval `scramv1 runtime -sh`
	set -x
	eval "cmsRun python/EfficiencyAnalysis_cfg.py sourceFileList=InputFiles/Era${1}${fileSuffix}.dat outputFileName=OutputFiles/Era${1}${fileSuffix}.root bunchSelection=NoSelection ${addJSON}"
	eval "cmsRun python/RefinedEfficiencyAnalysis_cfg.py sourceFileList=InputFiles/EraA${fileSuffix}.dat efficiencyFileName=OutputFiles/Era${1}.root outputFileName=OutputFiles/Era${1}_refinedEfficiency${fileSuffix}.root ${addJSON_A}"
	# eval "cmsRun python/InterpotEfficiency_cfg.py sourceFileList=InputFiles/Era${1}${fileSuffix}.dat outputFileName=OutputFiles/Era${1}_interpotEfficiency${fileSuffix}${outputFileSuffix}.root ${addJSON}"
	# eval "cmsRun python/EfficiencyVsXi_cfg.py era=${1} sourceFileList=InputFiles/EraA${fileSuffix}.dat efficiencyFileName=OutputFiles/Era${1}_refinedEfficiency${fileSuffix}.root outputFileName=OutputFiles/Era${1}_efficiencyVsXi${fileSuffix}.root ${addJSON_A}"
	# eval "cmsRun python/EfficiencyVsXi_cfg.py era=${1} sourceFileList=InputFiles/EraA${fileSuffix}.dat efficiencyFileName=OutputFiles/Era${1}_interpotEfficiency${fileSuffix}.root outputFileName=OutputFiles/Era${1}_efficiencyVsXi_fromMultiRP_XiSingle${fileSuffix}.root useMultiRPEfficiency=True ${addJSON_A}"
	set +x
fi