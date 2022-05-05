import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowClusterPS_cfi import *
from ..modules.particleFlowRecHitPS_cfi import *

pfClusteringPSTask = cms.Task(
    particleFlowClusterPS,
    particleFlowRecHitPS
)
