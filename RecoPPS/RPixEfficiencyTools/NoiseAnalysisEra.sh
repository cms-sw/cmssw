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
		# addJSON=" useJsonFile=true jsonFileName=/eos/project-c/ctpps/Operations/DataExternalConditions/2017/combined_RPIN_CMS_EraF2_TimingIn.json"
		# addJSON=" useJsonFile=true jsonFileName=/afs/cern.ch/user/a/abellora/Work/CT-PPS/2017_EfficiencyTool/CMSSW_10_6_10/src/RecoPPS/RPixEfficiencyTools/test/JSONFiles/EraB_simMultiRP_realDataCheck.json"
		
		fileSuffix="_6PlanesTracks"
	fi
	eval "cmsRun python/NoiseAnalysis_cfg.py sourceFileList=InputFiles/Era${1}.dat outputFileName=OutputFiles/Era${1}_noise${fileSuffix}.root maxPixelTracks=1 $addJSON"
fi