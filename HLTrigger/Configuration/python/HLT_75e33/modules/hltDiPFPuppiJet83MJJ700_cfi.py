import FWCore.ParameterSet.Config as cms

# At least one pair of jets with:
# - pT > 83.0 GeV;
# - |eta| < 5;
# - in opposite hemispheres;
# - Mjj > 700.0 GeV.
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_TkMu.
hltDiPFPuppiJet83MJJ700 = cms.EDFilter("HLTPFJetVBFFilter",
    etaOpposite = cms.bool(True),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    leadingJetOnly = cms.bool(False),
    maxEta = cms.double(5.0),
    minDeltaEta = cms.double(0.0),
    minInvMass = cms.double(700.0),
    minPtHigh = cms.double(83.0),
    minPtLow = cms.double(83.0),
    saveTags = cms.bool(True),
    triggerType = cms.int32(85)
)
