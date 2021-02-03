import FWCore.ParameterSet.Config as cms

from ..tasks.particleFlowClusterTask_cfi import *

particleFlowCluster = cms.Sequence(particleFlowClusterTask)
