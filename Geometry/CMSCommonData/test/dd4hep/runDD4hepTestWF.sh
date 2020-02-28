#!/bin/bash

set -x

# Check if CMSSW envs are setup
: ${CMSSW_BASE:?'You need to set CMSSW environemnt first.'}

# DEFAULTS

events=100
geometry2021='DD4hepExtended2021'

# ARGUMENT PARSING

while getopts ":n:" opt; do
  case $opt in
    n)
      echo "Generating $OPTARG events" >&1
      events=${OPTARG}
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# GEN-SIM goes first
if [ ! -f SingleMuPt10_pythia8_cfi_GEN_SIM.root ] ; then
  cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:phase1_2021_realistic \
--geometry ${geometry2021} \
--procModifiers dd4hep \
-n ${events} \
--era Run3 \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--beamspot NoSmear \
--nThreads=4 \
--fileout file:SingleMuPt10_pythia8_cfi_GEN_SIM.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM.py > SingleMuPt10_pythia8_cfi_GEN_SIM.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the GEN-SIM step, aborting."
      exit 1
    fi
fi
