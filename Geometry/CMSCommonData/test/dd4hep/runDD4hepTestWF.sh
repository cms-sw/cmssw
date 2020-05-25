#!/bin/sh -e

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
if [ ! -f step1.root ] ; then
  cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:phase1_2021_realistic \
--geometry ${geometry2021} \
-n ${events} \
--era Run3_dd4hep \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--beamspot NoSmear \
--nThreads=4 \
--fileout file:step1.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM.py > step1.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the GEN-SIM step, aborting."
      exit 1
    fi
fi

#DIGI-L1-HLT
if [ -f step1.root ] ; then
  cmsDriver.py step2 \
-s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2021 \
--conditions auto:phase1_2021_realistic \
--geometry ${geometry2021} \
-n ${events} \
--era Run3_dd4hep \
--eventcontent FEVTDEBUGHLT \
--datatier GEN-SIM-DIGI-RAW \
--nThreads=4 \
--filein  file:step1.root  \
--fileout file:step2.root > step2.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the DIGI-L1-HLT step, aborting."
      exit 1
    fi
fi

#RECO-DQM
if [ -f step2.root ] ; then
  cmsDriver.py step3  \
-s RAW2DIGI,L1Reco,RECO,RECOSIM,EI,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM \
--conditions auto:phase1_2021_realistic \
--geometry ${geometry2021} \
-n ${events} \
--era Run3_dd4hep \
--eventcontent RECOSIM,MINIAODSIM,DQM \
--datatier GEN-SIM-RECO,MINIAODSIM,DQMIO \
--runUnscheduled \
--nThreads=4 \
--filein  file:step2.root \
--fileout file:step3.root > step3.log 2>&1

    if [ $? -ne 0 ]; then
      echo "Error executing the RECO-DQM step, aborting."
      exit 1
    fi
fi
