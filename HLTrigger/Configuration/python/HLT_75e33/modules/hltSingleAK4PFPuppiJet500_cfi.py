import FWCore.ParameterSet.Config as cms

hltSingleAK4PFPuppiJet500 = cms.EDFilter("HLT1PFJet",
    MaxEta = cms.double(5.0),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(500.0),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(85)
)
