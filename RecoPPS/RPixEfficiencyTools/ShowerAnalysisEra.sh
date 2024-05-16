#!/bin/bash
if [ $# -ne 2 ]
then
    echo "Two arguments required. Nothing done."
    # ${1} -> Era
    # ${2} -> CMSSW_BASE
else
	nameTag=Run${1}
	export CMSSW_BASE=${2}
	export SCRAM_ARCH=slc7_amd64_gcc700
	export X509_USER_PROXY=$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/x509up_u$UID
	cd $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/
	eval `scramv1 runtime -sh`
	if true
	then	
		addJSON="useJsonFile=True jsonFileName=/afs/cern.ch/user/a/abellora/Work/CT-PPS/2018_EfficiencyTool/CMSSW_10_6_10/src/RecoPPS/RPixEfficiencyTools/test/JSONFiles/Run316569_Mid.json"
		fileSuffix="_Run316569_Mid"
	fi
	eval "cmsRun python/ShowerAnalysis_cfg.py sourceFileList=InputFiles/Era${1}_UL_AOD_Run316569.dat outputFileName=OutputFiles/Era${1}_showers${fileSuffix}.root $addJSON"
fi