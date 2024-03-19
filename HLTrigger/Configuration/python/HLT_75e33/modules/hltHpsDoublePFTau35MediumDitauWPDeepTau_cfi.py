import FWCore.ParameterSet.Config as cms

hltHpsDoublePFTau35MediumDitauWPDeepTau = cms.EDFilter("HLT1PFTau",
    MaxEta = cms.double(2.1),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(2),
    MinPt = cms.double(35.0),
    inputTag = cms.InputTag("hltHpsSelectedPFTausMediumDitauWPDeepTau"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(84)
)
