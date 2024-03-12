import FWCore.ParameterSet.Config as cms

gctInternJetProducer = cms.EDProducer("L1GctInternJetProducer",
    internalJetSource = cms.InputTag("gctDigis"),
    centralBxOnly = cms.bool(True)
)
# foo bar baz
# 4rYo8XsiDWs7d
# IgaTZRKzSK8fB
