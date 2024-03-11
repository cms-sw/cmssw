import FWCore.ParameterSet.Config as cms

hltHpsDoublePFTau40TrackPt1MediumChargedIsolation = cms.EDFilter("HLT1PFTau",
    MaxEta = cms.double(2.1),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(2),
    MinPt = cms.double(40.0),
    inputTag = cms.InputTag("hltHpsSelectedPFTausTrackPt1MediumChargedIsolation"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(84)
)
