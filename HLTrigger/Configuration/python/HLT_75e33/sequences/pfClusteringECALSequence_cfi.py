import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowClusterECALUncorrected_cfi import *
from ..modules.particleFlowRecHitECAL_cfi import *
from ..sequences.particleFlowClusterECALSequence_cfi import *

pfClusteringECALSequence = cms.Sequence(particleFlowRecHitECAL+particleFlowClusterECALUncorrected+particleFlowClusterECALSequence)
