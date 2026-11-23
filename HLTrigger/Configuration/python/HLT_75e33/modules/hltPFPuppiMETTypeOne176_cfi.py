import FWCore.ParameterSet.Config as cms

# For now, threshold aligned to the v0p0_2026
# L1 seed pDoubleJet_METSum.
hltPFPuppiMETTypeOne176 = cms.EDFilter("HLT1PFMET",
    MaxEta = cms.double(-1.0),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(176.0),
    inputTag = cms.InputTag("hltPFPuppiMETTypeOne"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(87)
)
