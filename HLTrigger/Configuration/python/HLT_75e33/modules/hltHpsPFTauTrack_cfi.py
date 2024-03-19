import FWCore.ParameterSet.Config as cms

hltHpsPFTauTrack = cms.EDFilter("HLT1PFTau",
    MaxEta = cms.double(2.5),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(0.0),
    inputTag = cms.InputTag("hltHpsPFTauProducer"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(84)
)
