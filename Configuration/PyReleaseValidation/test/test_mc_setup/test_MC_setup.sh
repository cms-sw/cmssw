#!/bin/bash

# This script mimics a MC chain with HLT run 
# a in a different release from the current one

echo '#### MC Setup'

workdir=$PWD
scriptdir=$CMSSW_BASE/src/Configuration/PyReleaseValidation/test/test_mc_setup/

# inputs
conditions=$1 
era=$2
hlt=$3
release_hlt=$4
gt_hlt=$5
beamspot=$6
fragment=$([ -n "$7" ] && echo "$7" || echo "TTbar_14TeV_TuneCP5_cfi")
era_reco=$([ -n "$8" ] && echo "$8" || echo "$era")

base_dir=$PWD
base_cms=$CMSSW_BASE
base_arch=$SCRAM_ARCH

# to take into account changes in gcc
base_arch_no_gcc=$(echo $SCRAM_ARCH | cut -d "_" -f -2)
if [[ ! ("$CMSSW_VERSION" == $release_hlt) ]]; then
    hlt_cmssw_path=$(scram list -c $release_hlt | grep -w $release_hlt | sed 's|.* ||')
    echo $hlt_cmssw_path
fi

echo '> GT           : ' $conditions
echo '> Era          : ' $era
echo '> BS           : ' $beamspot
echo '> HLT          : ' $hlt
echo '> HLT release  : ' $release_hlt
echo '> Fragment     : ' $fragment
echo '> Era at RECO  : ' $era_reco
if [[ ! ("$CMSSW_VERSION" == $release_hlt) ]]; then
    echo " - at ${hlt_cmssw_path}"
fi
echo '> HLT GT       : ' $gt_hlt
if [[ -z "${CMSSW_MC_SETUP_TEST_CATCH_HLT}" ]]; then
    echo ' - !! No error catch at HLT - If you want to catch them set  !!'
    echo ' - !! the environment variable CMSSW_MC_SETUP_TEST_CATCH_HLT !!'
fi
echo '> Working dir  : ' $workdir
echo ''
############################################################################################################
# GEN SIM

${scriptdir}/test_MC_setup_gen_sim.sh $CMSSW_VERSION $conditions $era $beamspot $fragment

if [ $? -ne 0 ]; then
    exit 1;
fi

############################################################################################################
# HLT
if [[ ! ("$CMSSW_VERSION" == $release_hlt) ]]; then
    cd $hlt_cmssw_path
    eval `scram runtime -sh`
    cd $workdir
fi

${scriptdir}/test_MC_setup_hlt.sh $gt_hlt $era $hlt
hlt_result=$?

if [ $hlt_result -ne 0 ] && [[ ! -z "${CMSSW_MC_SETUP_TEST_CATCH_HLT}" ]]; then
    exit 1;
elif [ $hlt_result -ne 0 ]; then
    echo "!!! HLT failed but ignoring !!!"
    exit 0
fi

############################################################################################################
# RECO + PAT
if [[ ! ("$CMSSW_BASE" == $base_cms) ]]; then
    cd $base_cms
    eval `scram runtime -sh`
    cd $workdir
fi
echo $SCRAM_ARCH
${scriptdir}/test_MC_setup_reco.sh $release_hlt $conditions $era_reco

if [ $? -ne 0 ]; then
    exit 1;
fi

############################################################################################################

echo '>>>> Done! <<<<'
