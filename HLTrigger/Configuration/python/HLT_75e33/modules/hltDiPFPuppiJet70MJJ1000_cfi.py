import FWCore.ParameterSet.Config as cms

# At least one pair of jets with:
# - pT > 70.0 GeV;
# - |eta| < 5;
# - in opposite hemispheres;
# - Mjj > 1000.0 GeV.
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_Mass_Min1000.
hltDiPFPuppiJet70MJJ1000 = cms.EDFilter("HLTPFJetVBFFilter",
    etaOpposite = cms.bool(True),
    inputTag = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    leadingJetOnly = cms.bool(False),
    maxEta = cms.double(5.0),
    minDeltaEta = cms.double(0.0),
    minInvMass = cms.double(1000.0),
    minPtHigh = cms.double(70.0),
    minPtLow = cms.double(70.0),
    saveTags = cms.bool(True),
    triggerType = cms.int32(85)
)
