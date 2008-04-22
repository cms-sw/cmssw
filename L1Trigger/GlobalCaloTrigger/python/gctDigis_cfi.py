import FWCore.ParameterSet.Config as cms

gctDigis = cms.EDFilter("L1GctEmulator",
    jetFinderType = cms.string('hardwareJetFinder'),
    inputLabel = cms.InputTag("rctDigis"),
    preSamples = cms.uint32(2),
    postSamples = cms.uint32(2)
)


