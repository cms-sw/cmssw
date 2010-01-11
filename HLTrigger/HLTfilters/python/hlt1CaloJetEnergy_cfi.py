import FWCore.ParameterSet.Config as cms

hlt1CaloJetEnergy = cms.EDFilter("HLT1CaloJetEnergy",
    saveTag = cms.untracked.bool( False ),
    inputTag = cms.InputTag("iterativeCone5CaloJets"),
    MinE = cms.double(30.0),
    MaxEta = cms.double(3.0),
    MinN = cms.int32(1)
)
