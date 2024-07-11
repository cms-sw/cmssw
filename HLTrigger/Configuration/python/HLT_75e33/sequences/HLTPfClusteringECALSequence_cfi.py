import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterECALUncorrected_cfi import *
from ..modules.hltParticleFlowRecHitECAL_cfi import *
from ..sequences.HLTParticleFlowClusterECALSequence_cfi import *

HLTPfClusteringECALSequence = cms.Sequence(hltParticleFlowRecHitECAL+hltParticleFlowClusterECALUncorrected+HLTParticleFlowClusterECALSequence)
