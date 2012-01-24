import FWCore.ParameterSet.Config as cms

hltNVFilter = cms.EDFilter("HLTNVFilter",
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTags = cms.bool( False ),
    minEtJet2 = cms.double(20.0),
    minEtJet1 = cms.double(80.0),
    minNV = cms.double(0.1),
    inputMETTag = cms.InputTag("hlt1MET60")
)


