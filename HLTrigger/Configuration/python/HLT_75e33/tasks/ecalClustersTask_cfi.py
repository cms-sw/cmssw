import FWCore.ParameterSet.Config as cms

from ..tasks.ecalClustersNoPFBoxTask_cfi import *
from ..tasks.particleFlowSuperClusteringTask_cfi import *

ecalClustersTask = cms.Task(ecalClustersNoPFBoxTask, particleFlowSuperClusteringTask)
