import FWCore.ParameterSet.Config as cms

hlt1CaloJetEnergy = cms.EDFilter("HLT1CaloJetEnergy",
                                 inputTag = cms.InputTag("iterativeCone5CaloJets"),
                                 MinE = cms.double(30.0),
                                 MaxEta = cms.double(3.0),
                                 MinN = cms.int32(1)
)
