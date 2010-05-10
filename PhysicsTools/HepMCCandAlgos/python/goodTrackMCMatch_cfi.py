import FWCore.ParameterSet.Config as cms

goodTrackMCMatch = cms.EDProducer("MCMatcher",
    src = cms.InputTag("goodTracks"),
    maxDPtRel = cms.double(1.0),
    mcPdgId = cms.vint32(13), ## muons

    mcStatus = cms.vint32(1),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.15),
    checkCharge = cms.bool(True),
    resolveAmbiguities = cms.bool(False),
    matched = cms.InputTag("genParticles")
)


