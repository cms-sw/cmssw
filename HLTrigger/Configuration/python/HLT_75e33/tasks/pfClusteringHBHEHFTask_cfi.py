import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowClusterHBHE_cfi import *
from ..modules.particleFlowClusterHCAL_cfi import *
from ..modules.particleFlowClusterHF_cfi import *
from ..modules.particleFlowRecHitHBHE_cfi import *
from ..modules.particleFlowRecHitHF_cfi import *

pfClusteringHBHEHFTask = cms.Task(
    particleFlowClusterHBHE,
    particleFlowClusterHCAL,
    particleFlowClusterHF,
    particleFlowRecHitHBHE,
    particleFlowRecHitHF
)
