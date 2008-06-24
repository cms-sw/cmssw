import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the MC match
# for taus (cuts are NOT tuned)
# (using old values from TQAF, january 2008)
#
tauMatch = cms.EDFilter("PATMCMatcher",
    src = cms.InputTag("allLayer0Taus"),    # RECO objects to match
    matched = cms.InputTag("genParticles"), # mc-truth particle collection
    mcPdgId     = cms.vint32(15), # one or more PDG ID (15 = tau); absolute values (see below)
    checkCharge = cms.bool(True), # True = require RECO and MC objects to have the same charge
    mcStatus = cms.vint32(2),     # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
                                  # NOTE that Taus can only be status 3 or 2, never 1!
    maxDeltaR = cms.double(999.9), # Minimum deltaR for the match.     By default any deltaR is allowed (why??)
    maxDPtRel = cms.double(999.9), # Minimum deltaPt/Pt for the match. By default anything is allowed   ( ""  )
    resolveAmbiguities = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False), # False = just match input in order; True = pick lowest deltaR pair first
)
