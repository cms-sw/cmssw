import FWCore.ParameterSet.Config as cms

from DQM.PFTasks.pfHcalGPUComparisonTask_cfi import *

pfClusterHBHEOnlyAlpakaComparison = pfHcalGPUComparisonTask.clone(
    compDir = cms.untracked.string('HeterogeneousComparisons/ParticleFlow'),
    pfClusterToken_ref = cms.untracked.InputTag('particleFlowClusterHBHEOnlyLegacy'),
    pfClusterToken_target = cms.untracked.InputTag('particleFlowClusterHBHEOnly'),
)
