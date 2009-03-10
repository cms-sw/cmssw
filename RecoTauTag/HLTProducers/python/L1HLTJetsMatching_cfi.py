import FWCore.ParameterSet.Config as cms

l1HLTTauJetsMatching = cms.EDFilter("L1HLTTauJetsMatching",
    L1TauTrigger = cms.InputTag("DummyHLTL1SeedFilter"),
    EtMin = cms.double(15.0),
    JetSrc = cms.InputTag("hltL2TauJets")
)


