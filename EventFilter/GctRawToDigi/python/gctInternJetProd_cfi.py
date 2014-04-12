import FWCore.ParameterSet.Config as cms

gctInternJetProducer = cms.EDProducer("L1GctInternJetProducer",
    internalJetSource = cms.InputTag("gctDigis"),
    centralBxOnly = cms.bool(True)
)
