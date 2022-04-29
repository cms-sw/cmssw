import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowClusterECALUncorrected_cfi import *
from ..modules.particleFlowRecHitECAL_cfi import *
from ..tasks.particleFlowClusterECALTask_cfi import *

pfClusteringECALTask = cms.Task(
    particleFlowClusterECALTask,
    particleFlowClusterECALUncorrected,
    particleFlowRecHitECAL
)
