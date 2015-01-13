import FWCore.ParameterSet.Config as cms

gctDigis = cms.EDProducer("L1GctEmulator",
    jetFinderType = cms.string('hardwareJetFinder'),
    hardwareTest = cms.bool(False),
    writeInternalData = cms.bool(False),
    useImprovedTauAlgorithm = cms.bool(True),
    ignoreRCTTauVetoBitsForIsolation = cms.bool(False),
    inputLabel = cms.InputTag("rctDigis"),
    preSamples = cms.uint32(1),
    postSamples = cms.uint32(1),
    conditionsLabel = cms.string("")
)


  
