import FWCore.ParameterSet.Config as cms

electronMatch = cms.EDFilter("PATMCMatcher",
    src = cms.InputTag("allLayer0Electrons"),
    maxDPtRel = cms.double(0.5),
    mcPdgId = cms.vint32(11),
    mcStatus = cms.vint32(1),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.5),
    checkCharge = cms.bool(True),
    resolveAmbiguities = cms.bool(True),
    matched = cms.InputTag("genParticles")
)


