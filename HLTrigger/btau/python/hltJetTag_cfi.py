import FWCore.ParameterSet.Config as cms

hltJetTag = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("taggedJetCollection"),
    MinTag = cms.double(2.0),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(False),
    MinJets = cms.int32(1)
)


