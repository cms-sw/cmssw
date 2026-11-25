import FWCore.ParameterSet.Config as cms

# At least one pair of jets with:
# - pT > 93.0 GeV;
# - |eta| < 5;
# - in opposite hemispheres;
# - Mjj > 600.0 GeV.
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_IsoTau.
hltDiPFPuppiJet93MJJ600 = cms.EDFilter("HLTPFJetVBFFilter",
    etaOpposite = cms.bool(True),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    leadingJetOnly = cms.bool(False),
    maxEta = cms.double(5.0),
    minDeltaEta = cms.double(0.0),
    minInvMass = cms.double(600.0),
    minPtHigh = cms.double(93.0),
    minPtLow = cms.double(93.0),
    saveTags = cms.bool(True),
    triggerType = cms.int32(85)
)
