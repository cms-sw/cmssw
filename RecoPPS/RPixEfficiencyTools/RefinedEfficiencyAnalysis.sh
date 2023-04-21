#!/bin/bash
if [ $# -ne 2 ]
then
    echo "Two arguments required. Nothing done."
else
	export CMSSW_BASE=${2}
	export SCRAM_ARCH=slc7_amd64_gcc700
	export X509_USER_PROXY=$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/x509up_u93252
	cd $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/
	eval `scramv1 runtime -sh`
	eval "cmsRun python/RefinedEfficiencyAnalysis_cfg.py sourceFileList=InputFiles/Run${1}.dat efficiencyFileName=OutputFiles/Run${1}.root runNumber=${1}"
fi