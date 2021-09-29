import FWCore.ParameterSet.Config as cms

hltPFPuppiJetForBtagSelectorEta2p4 = cms.EDFilter("HLT1PFJet",
    MaxEta = cms.double(2.4),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-2.4),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(30.0),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(86)
)
