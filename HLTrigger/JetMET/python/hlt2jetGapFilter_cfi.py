import FWCore.ParameterSet.Config as cms

hlt2jetGapFilter = cms.EDFilter("HLT2jetGapFilter",
    minEt = cms.double(90.0),
    inputTag = cms.InputTag("iterativeCone5CaloJets"),
    minEta = cms.double(1.9)
)


