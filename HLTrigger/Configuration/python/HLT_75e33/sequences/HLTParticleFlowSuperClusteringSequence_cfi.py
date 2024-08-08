import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowSuperClusterECAL_cfi import *

HLTParticleFlowSuperClusteringSequence = cms.Sequence(hltParticleFlowSuperClusterECAL)
