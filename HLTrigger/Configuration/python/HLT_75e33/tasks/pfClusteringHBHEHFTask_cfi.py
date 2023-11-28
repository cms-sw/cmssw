import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterHBHE_cfi import *
from ..modules.hltParticleFlowClusterHCAL_cfi import *
from ..modules.particleFlowClusterHF_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *
from ..modules.particleFlowRecHitHF_cfi import *

pfClusteringHBHEHFTask = cms.Task(
    hltParticleFlowClusterHBHE,
    hltParticleFlowClusterHCAL,
    particleFlowClusterHF,
    hltParticleFlowRecHitHBHE,
    particleFlowRecHitHF
)
