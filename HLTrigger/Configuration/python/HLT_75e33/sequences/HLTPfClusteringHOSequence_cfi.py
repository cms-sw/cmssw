import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterHO_cfi import *
from ..modules.hltParticleFlowRecHitHO_cfi import *

HLTPfClusteringHOSequence = cms.Sequence(hltParticleFlowRecHitHO+hltParticleFlowClusterHO)
