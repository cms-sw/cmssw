import FWCore.ParameterSet.Config as cms

from DQMOffline.ParticleFlow.pfHcalGPUComparisonTask_cfi import pfHcalGPUComparisonTask

pfClusterHBHEOnlyAlpakaComparison = pfHcalGPUComparisonTask.clone(
    compDir = cms.untracked.string('HeterogeneousComparisons/ParticleFlow/HCAL'),
    pfClusterToken_ref = cms.untracked.InputTag('particleFlowClusterHBHEOnlyLegacy'),
    pfClusterToken_target = cms.untracked.InputTag('particleFlowClusterHBHEOnly'),
)
