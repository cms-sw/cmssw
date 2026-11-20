import FWCore.ParameterSet.Config as cms

# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_METSum.
hltSinglePFPuppiJet118MaxEta5 = cms.EDFilter("HLT1PFJet",
    MaxEta = cms.double(5.0),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(118.0),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(85)
)
