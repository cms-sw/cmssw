import FWCore.ParameterSet.Config as cms

# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_Mass_Min1000.
hltDoublePFPuppiJets70MaxEta5p2 = cms.EDFilter("HLT1PFJet",
    MaxEta = cms.double(5.2),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(2),
    MinPt = cms.double(70.0),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(85)
)
