import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterHBHE_cfi import *
from ..modules.hltParticleFlowClusterHCAL_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *

HLTPFHcalClusteringForEgammaTask = cms.Task(
    hltParticleFlowClusterHBHE,
    hltParticleFlowClusterHCAL,
    hltParticleFlowRecHitHBHE
)
