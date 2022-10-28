import FWCore.ParameterSet.Config as cms

hltPFSoftKillerMET = cms.EDProducer("PFMETProducer",
    calculateSignificance = cms.bool(False),
    globalThreshold = cms.double(0.0),
    src = cms.InputTag("hltParticleFlowSoftKiller")
)
