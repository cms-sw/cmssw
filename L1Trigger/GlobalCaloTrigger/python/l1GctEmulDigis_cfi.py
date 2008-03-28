import FWCore.ParameterSet.Config as cms

l1GctEmulDigis = cms.EDFilter("L1GctEmulator",
    jetFinderType = cms.string('hardwareJetFinder'),
    inputLabel = cms.InputTag("l1RctEmulDigis")
)


