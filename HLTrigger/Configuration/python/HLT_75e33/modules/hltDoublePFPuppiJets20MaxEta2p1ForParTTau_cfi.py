import FWCore.ParameterSet.Config as cms

# Prefilter before the ParT sequence:
# >= 2 uncorrected PUPPI AK4 jets with pT > 20 GeV & |eta| < 2.1.
hltDoublePFPuppiJets20MaxEta2p1ForParTTau = cms.EDFilter("HLT1PFJet",
    MaxEta = cms.double(2.1),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(2),
    MinPt = cms.double(20.0),
    inputTag = cms.InputTag("hltAK4PFPuppiJets"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(86)
)
