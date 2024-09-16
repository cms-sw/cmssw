import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterHBHE_cfi import *
from ..modules.hltParticleFlowClusterHCAL_cfi import *
from ..modules.hltParticleFlowRecHitHBHE_cfi import *
from ..modules.hltParticleFlowClusterHF_cfi import *
from ..modules.hltParticleFlowRecHitHF_cfi import *

HLTPfClusteringHBHEHFSequence = cms.Sequence(hltParticleFlowRecHitHBHE+hltParticleFlowClusterHBHE+hltParticleFlowClusterHCAL+hltParticleFlowRecHitHF+hltParticleFlowClusterHF)
