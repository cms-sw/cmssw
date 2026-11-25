import FWCore.ParameterSet.Config as cms

# One ParT tau-tagged uncorrected PUPPI AK4 jet with pT > 61 GeV and |eta| < 2.1, at the
# Medium working point of HLT_DoubleMediumPFPuppiParTTauh30_eta2p1.
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_IsoTau.
hltSinglePFPuppiJet61ParTTauhTagMediumWP = cms.EDFilter("TauTagFilter",
    matchWithSeeds = cms.bool(False),
    matchingdR = cms.double(0.5),
    maxEta = cms.double(2.1),
    minPt = cms.double(61.0),
    nExpected = cms.int32(1),
    saveTags = cms.bool(True),
    seedTypes = cms.vint32(84),
    seeds = cms.InputTag(""),
    selection = cms.string('0.11645'),
    tauPtCorr = cms.InputTag(""),
    tauTags = cms.InputTag("hltParticleTransformerDiscriminatorsJetTags","TauvsAll"),
    taus = cms.InputTag("hltAK4PFPuppiJets"),
    usePtCorr = cms.bool(False)
)
