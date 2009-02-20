import FWCore.ParameterSet.Config as cms

gctDigis = cms.EDFilter("L1GctEmulator",
    jetFinderType = cms.string('hardwareJetFinder'),
    useImprovedTauAlgorithm = cms.bool(False),
    ignoreRCTTauVetoBitsForIsolation = cms.bool(False),
    inputLabel = cms.InputTag("rctDigis"),
    preSamples = cms.uint32(1),
    postSamples = cms.uint32(1)
)


  
