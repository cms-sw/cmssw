import FWCore.ParameterSet.Config as cms

from ..tasks.particleFlowSuperClusteringTask_cfi import *

ecalClustersTask = cms.Task(
    particleFlowSuperClusteringTask
)
