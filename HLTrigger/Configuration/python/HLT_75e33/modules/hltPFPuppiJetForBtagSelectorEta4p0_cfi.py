import FWCore.ParameterSet.Config as cms

hltPFPuppiJetForBtagSelectorEta4p0 = cms.EDFilter("HLT1PFJet",
    MaxEta = cms.double(4.0),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-4.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(30.0),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(86)
)
