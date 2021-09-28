#!/bin/bash
if [ $# -ne 2 ]
then
    echo "Two arguments required. Nothing done."
    # ${1} -> Run Number
    # ${2} -> CMSSW_BASE
else
	nameTag=Run${1}
	export CMSSW_BASE=${2}
	export SCRAM_ARCH=slc7_amd64_gcc700
	export X509_USER_PROXY=$CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/x509up_u$UID
	cd $CMSSW_BASE/src/RecoPPS/RPixEfficiencyTools/
	eval `scramv1 runtime -sh`
	eval "cmsRun python/NoiseAnalysis_cfg.py sourceFileList=InputFiles/${nameTag}.dat outputFileName=OutputFiles/${nameTag}_noise.root runNumber=${1} recoInfo=-1 maxPixelTracks=99"
fi