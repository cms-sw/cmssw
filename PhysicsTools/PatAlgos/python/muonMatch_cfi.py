import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the MC match
# for muons (cuts are NOT tuned)
# (using old values from TQAF, january 2008)
#
muonMatch = cms.EDFilter("PATMCMatcher",
    src = cms.InputTag("allLayer0Muons"),
    maxDPtRel = cms.double(0.5),
    mcPdgId = cms.vint32(13),
    mcStatus = cms.vint32(1),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    checkCharge = cms.bool(True),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("genParticles")
)


