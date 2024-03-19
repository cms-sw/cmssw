import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowClusterHO_cfi import *
from ..modules.particleFlowRecHitHO_cfi import *

pfClusteringHOSequence = cms.Sequence(particleFlowRecHitHO+particleFlowClusterHO)
