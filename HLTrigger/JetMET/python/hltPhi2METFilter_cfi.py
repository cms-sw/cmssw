import FWCore.ParameterSet.Config as cms

hltPhi2METFilter = cms.EDFilter("HLTPhi2METFilter",
    saveTags = cms.bool( False ),
    maxDeltaPhi = cms.double(3.1514),
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    inputMETTag = cms.InputTag("hlt1MET60"),
    minDeltaPhi = cms.double(0.377),
    minEtJet1 = cms.double(60.0),
    minEtJet2 = cms.double(60.0)
)


