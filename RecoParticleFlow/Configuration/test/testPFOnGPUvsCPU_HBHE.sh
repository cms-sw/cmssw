#!/bin/bash -e

[ ! -z "${CMSSW_BASE}" ] || exit 1

cmsRun "${CMSSW_BASE}"/src/RecoParticleFlow/Configuration/test/run_HBHEandPF_onCPUandGPU.py
cmsRun "${CMSSW_BASE}"/src/Validation/RecoParticleFlow/test/run_DQM_forGPU_HLT.py
cmsRun "${CMSSW_BASE}"/src/Validation/RecoParticleFlow/test/run_HARVESTING_forGPU_HLT.py &> /dev/null
