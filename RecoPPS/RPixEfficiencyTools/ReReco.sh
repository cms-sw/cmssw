#!/bin/bash
if [ $# -ne 5 ]
then
    echo "Five arguments required. Nothing done."
else
	export CMSSW_BASE=${2}
	export SCRAM_ARCH=slc7_amd64_gcc700
	# export X509_USER_PROXY=$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/x509up_u93252
	cd $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test
	eval `scramv1 runtime -sh`
	eval "cmsRun $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/test/ReReco.py runNumber=${1} skipEvents=${3} maxEvents=${4} \
	outputFileName=/eos/project/c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2018/ReRecoOutputTmp_CMSSW_10_6_2/Run${1}_${5}.root"
fi