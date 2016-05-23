import FWCore.ParameterSet.Config as cms

process = cms.Process("Layer1Validator")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('L1Trigger.L1TCaloLayer1.layer1Validator_cfi')
process.layer1Validator.testSource = cms.InputTag("l1tCaloLayer1SpyDigis")
process.layer1Validator.emulSource = cms.InputTag("layer1EmulatorDigis")
process.layer1Validator.verbose = cms.bool(True)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/dasu/l1tCaloLayer1Spy+Emulator.root')
)

process.p = cms.Path(process.layer1Validator)

