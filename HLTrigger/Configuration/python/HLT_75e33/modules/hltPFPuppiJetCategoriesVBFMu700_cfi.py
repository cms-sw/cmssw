import FWCore.ParameterSet.Config as cms

# VBF jet selection:
# among jets with pT > minPt2, take the pair with the largest Mjj and require:
# - Mjj > minInvMass and:
#   - either its harder jet with pT >= minPt1 (2-jet category);
#   - or a third jet with pT > minPt3 (3-jet category).
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_TkMu.
hltPFPuppiJetCategoriesVBFMu700 = cms.EDFilter("HLTL1TMatchedPFJetsVBFFilter",
    algorithm = cms.string('VBF'),
    l1tJetRefs = cms.InputTag("hltL1SeedsForVBFDiPFPuppiJetFilter"),
    matchJetsByDeltaR = cms.bool(False),
    maxJetDeltaR = cms.double(0.5),
    maxNJets = cms.int32(-1),
    minInvMass = cms.double(700.0),
    minNJets = cms.int32(2),
    minPt1 = cms.double(133.0),
    minPt2 = cms.double(83.0),
    minPt3 = cms.double(133.0),
    saveTags = cms.bool(True),
    src = cms.InputTag("hltAK4PFPuppiJetsCorrected"),
    triggerType = cms.int32(85)
)
