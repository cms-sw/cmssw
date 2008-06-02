import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the MC match
# for photons (cuts are NOT tuned)
# (using old values from TQAF, january 2008)
#
photonMatch = cms.EDFilter("PATMCMatcher",
    src = cms.InputTag("allLayer0Photons"),
    maxDPtRel = cms.double(1.0),
    mcPdgId = cms.vint32(22),
    mcStatus = cms.vint32(1),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.2),
    checkCharge = cms.bool(True),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("genParticles")
)


