import FWCore.ParameterSet.Config as cms

simGctDigis = cms.EDProducer("L1GctEmulator",
    conditionsLabel = cms.string(''),
    hardwareTest = cms.bool(False),
    ignoreRCTTauVetoBitsForIsolation = cms.bool(False),
    inputLabel = cms.InputTag("simRctDigis"),
    jetFinderType = cms.string('hardwareJetFinder'),
    postSamples = cms.uint32(1),
    preSamples = cms.uint32(1),
    useImprovedTauAlgorithm = cms.bool(True),
    writeInternalData = cms.bool(False)
)
