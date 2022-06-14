import FWCore.ParameterSet.Config as cms

hltHpsDoublePFTau22 = cms.EDFilter("HLT1PFTau",
    MaxEta = cms.double(2.1),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-2.1),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(2),
    MinPt = cms.double(22.0),
    inputTag = cms.InputTag("hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(84)
)
