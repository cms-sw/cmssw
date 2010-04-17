import FWCore.ParameterSet.Config as cms

myMuonMatch = cms.EDProducer("MCMatcher",     # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src     = cms.InputTag("muons"),        # RECO objects to match
    matched = cms.InputTag("genParticles"), # mc-truth particle collection
    mcPdgId     = cms.vint32(13),           # one or more PDG ID (13 = muon); absolute values (see below)
    checkCharge = cms.bool(True),           # True = require RECO and MC objects to have the same charge
    mcStatus = cms.vint32(1),               # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR = cms.double(0.5),            # Minimum deltaR for the match
    maxDPtRel = cms.double(0.5),            # Minimum deltaPt/Pt for the match
    resolveAmbiguities = cms.bool(True),    # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False) # False = just match input in order; True = pick lowest deltaR pair first
)
myJetGenJetMatch = cms.EDProducer("GenJetMatcher", # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src      = cms.InputTag("ak5CaloJets"), # RECO jets (any View<Jet> is ok)
    matched  = cms.InputTag("ak5GenJets"),  # GEN jets  (must be GenJetCollection)
    mcPdgId  = cms.vint32(),                # n/a
    mcStatus = cms.vint32(),                # n/a
    checkCharge = cms.bool(False),          # n/a
    maxDeltaR = cms.double(0.4),            # Minimum deltaR for the match
    maxDPtRel = cms.double(3.0),            # Minimum deltaPt/Pt for the match
    resolveAmbiguities = cms.bool(True),    # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False) # False = just match input in order; True = pick lowest deltaR pair first
)
  
