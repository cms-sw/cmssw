import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the MC match
# for electrons
#
electronMatch = cms.EDProducer("MCMatcher",       # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = cms.InputTag("gedGsfElectrons"), # RECO objects to match
    matched     = cms.InputTag("genParticles"), # mc-truth particle collection
    mcPdgId     = cms.vint32(11),               # one or more PDG ID (11 = electron); absolute values (see below)
    checkCharge = cms.bool(True),               # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.5),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),    # False = just match input in order; True = pick lowest deltaR pair first
)


