import FWCore.ParameterSet.Config as cms

# Prefilter before the ParT sequence: at least one uncorrected PUPPI AK4 jet with pT > 61 GeV
# and |eta| < 2.1.
hltSinglePFPuppiJet61MaxEta2p1ForParTTau = cms.EDFilter("HLT1PFJet",
    MaxEta = cms.double(2.1),
    MaxMass = cms.double(-1.0),
    MinE = cms.double(-1.0),
    MinEta = cms.double(-1.0),
    MinMass = cms.double(-1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(61.0),
    inputTag = cms.InputTag("hltAK4PFPuppiJets"),
    saveTags = cms.bool(True),
    triggerType = cms.int32(86)
)
