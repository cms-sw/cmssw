import FWCore.ParameterSet.Config as cms

hltPFPuppiMETTypeOne140 = cms.EDFilter("HLT1PFMET",
    MaxEta = cms.double(-1.0),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(140.0),
    inputTag = cms.InputTag("hltPFPuppiMETTypeOne"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(87)
)
