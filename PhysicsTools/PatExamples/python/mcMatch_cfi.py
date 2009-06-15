import FWCore.ParameterSet.Config as cms

myMuonMatch = cms.EDFilter(  ,                   # MC matcher module instance
    src                   = cms.InputTag(  ),    # RECO objects to match
    matched               = cms.InputTag(  ),    # mc-truth particle collection
    mcPdgId               = cms.vint32(  ),      # one or more PDG ID; absolute values (see below)
    checkCharge           = cms.bool(  ),        # True = require RECO and MC objects to have the same charge
    mcStatus              = cms.vint32(  ),      # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR             = cms.double(  ),      # Minimum deltaR for the match
    maxDPtRel             = cms.double(  ),      # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(  ),        # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(  )         # False = just match input in order; True = pick lowest deltaR pair first
)
myJetGenJetMatch = cms.EDFilter(  ,                        # MC matcher module instance
    src                   = cms.InputTag(  ),     # RECO jets (any View<Jet> is ok)
    matched               = cms.InputTag(  ),     # GEN jets  (must be GenJetCollection)
    mcPdgId               = cms.vint32(  ),       # n/a
    mcStatus              = cms.vint32(  ),       # n/a
    checkCharge           = cms.bool(  ),         # n/a
    maxDeltaR             = cms.double(  ),       # Minimum deltaR for the match
    maxDPtRel             = cms.double(  ),       # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(  ),         # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(  )          # False = just match input in order; True = pick lowest deltaR pair first
)
