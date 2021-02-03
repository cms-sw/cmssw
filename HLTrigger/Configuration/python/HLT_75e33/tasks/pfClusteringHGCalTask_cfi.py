import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowRecHitHGC_cfi import *

pfClusteringHGCalTask = cms.Task(particleFlowRecHitHGC)
