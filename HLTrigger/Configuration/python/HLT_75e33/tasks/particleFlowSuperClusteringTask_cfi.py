import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowSuperClusterECAL_cfi import *
from ..modules.particleFlowSuperClusterHGCal_cfi import *
from ..modules.particleFlowSuperClusterHGCalFromMultiCl_cfi import *

particleFlowSuperClusteringTask = cms.Task(
    particleFlowSuperClusterECAL,
    particleFlowSuperClusterHGCal,
    particleFlowSuperClusterHGCalFromMultiCl
)
