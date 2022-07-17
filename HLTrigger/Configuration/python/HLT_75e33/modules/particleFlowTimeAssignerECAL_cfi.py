import FWCore.ParameterSet.Config as cms

particleFlowTimeAssignerECAL = cms.EDProducer("PFClusterTimeAssigner",
    mightGet = cms.optional.untracked.vstring,
    src = cms.InputTag("particleFlowClusterECALUncorrected"),
    timeResoSrc = cms.InputTag("ecalBarrelClusterFastTimer","PerfectResolutionModelResolution"),
    timeSrc = cms.InputTag("ecalBarrelClusterFastTimer","PerfectResolutionModel")
)
