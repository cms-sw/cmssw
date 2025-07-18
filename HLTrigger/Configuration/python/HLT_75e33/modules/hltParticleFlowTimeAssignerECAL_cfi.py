import FWCore.ParameterSet.Config as cms

hltParticleFlowTimeAssignerECAL = cms.EDProducer("PFClusterTimeAssigner",
    mightGet = cms.optional.untracked.vstring,
    src = cms.InputTag("hltParticleFlowClusterECALUncorrected"),
    timeResoSrc = cms.InputTag("hltEcalBarrelClusterFastTimer","PerfectResolutionModelResolution"),
    timeSrc = cms.InputTag("hltEcalBarrelClusterFastTimer","PerfectResolutionModel")
)
