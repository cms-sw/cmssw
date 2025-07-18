import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterECALUncorrected_cfi import *
from ..sequences.HLTParticleFlowClusterECALSequence_cfi import *

HLTPfClusteringECALSequence = cms.Sequence(hltParticleFlowClusterECALUncorrected+HLTParticleFlowClusterECALSequence)
