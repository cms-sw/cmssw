import FWCore.ParameterSet.Config as cms

tauMatch = cms.EDFilter("PATMCMatcher",
    src = cms.InputTag("allLayer0Taus"),
    maxDPtRel = cms.double(99.0),
    mcPdgId = cms.vint32(15),
    mcStatus = cms.vint32(2),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(5.0),
    checkCharge = cms.bool(True),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("genParticles")
)


