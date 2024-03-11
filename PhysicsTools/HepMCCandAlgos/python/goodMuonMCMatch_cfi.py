import FWCore.ParameterSet.Config as cms

goodMuonMCMatch = cms.EDProducer("MCMatcher",
    src = cms.InputTag("goodMuons"),
    maxDPtRel = cms.double(1.0),
    mcPdgId = cms.vint32(13), ## muons

    mcStatus = cms.vint32(1),
    resolveByMatchQuality = cms.bool(False),
    maxDeltaR = cms.double(0.15),
    checkCharge = cms.bool(True),
    resolveAmbiguities = cms.bool(False),
    matched = cms.InputTag("genParticles")
)


# foo bar baz
