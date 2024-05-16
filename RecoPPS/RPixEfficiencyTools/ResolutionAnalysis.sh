#!/bin/bash
if [ $# -ne 2 ]
then
    echo "Two arguments required. Nothing done."
else
	export CMSSW_BASE=${2}
	export SCRAM_ARCH=slc6_amd64_gcc530
	export X509_USER_PROXY=$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/x509up_u93252
	cd $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/
	eval `scramv1 runtime -sh`
	eval "cmsRun python/ResolutionAnalysis_cfg.py sourceFileList=InputFiles/Run${1}.dat outputFileName=OutputFiles/Resolution/ResolutionRun${1}.root \
	runNumber=${1} minPointsForFit=3 maxPointsForFit=6 minNumberOfCls2=0 maxNumberOfCls2=6"
fi
