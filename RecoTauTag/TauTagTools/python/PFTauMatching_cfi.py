import FWCore.ParameterSet.Config as cms

pfTauMatcher = cms.EDProducer(
    "PFTauMatcher",
    src = cms.InputTag("your_collection"),
    matched = cms.InputTag("hpsTancTaus"),
    mcPdgId     = cms.vint32(),                      # n/a
    mcStatus    = cms.vint32(),                      # n/a
    checkCharge = cms.bool(True),                    # Require charge is correct
    maxDeltaR   = cms.double(0.15),                  # Minimum deltaR for the match
    maxDPtRel   = cms.double(3.0),                   # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(False),         # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),         # False = just match input in order; True = pick lowest deltaR pair first
)

candViewMatchedToPFTauRefSelector = cms.EDFilter(
    "CandViewPFTauMatchRefSelector",
    src = cms.InputTag("kinematicSignalJets"),
    matching = cms.InputTag("pfTauMatcher"),
    # keep events with no matches
    filter = cms.bool(False)
)
