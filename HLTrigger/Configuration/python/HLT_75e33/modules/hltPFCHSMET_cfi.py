import FWCore.ParameterSet.Config as cms

hltPFCHSMET = cms.EDProducer("PFMETProducer",
    calculateSignificance = cms.bool(False),
    globalThreshold = cms.double(0.0),
    src = cms.InputTag("hltParticleFlowCHS")
)
