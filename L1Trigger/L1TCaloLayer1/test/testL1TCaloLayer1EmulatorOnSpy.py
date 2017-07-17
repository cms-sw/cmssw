import FWCore.ParameterSet.Config as cms

process = cms.Process("Layer1EmulatorOnSpy")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_condDBv2_cff')
process.GlobalTag.globaltag = '74X_dataRun2_Express_v1'

process.load('L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi')
process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("l1tCaloLayer1SpyDigis")
process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("l1tCaloLayer1SpyDigis")
process.simCaloStage2Layer1Digis.useECALLUT = cms.bool(False)
process.simCaloStage2Layer1Digis.useHCALLUT = cms.bool(False)
process.simCaloStage2Layer1Digis.useHFLUT = cms.bool(False)
process.simCaloStage2Layer1Digis.useLSB = cms.bool(False)
process.simCaloStage2Layer1Digis.verbose = cms.bool(True)

process.load('L1Trigger.L1TCaloLayer1.layer1Validator_cfi')
process.layer1Validator.testSource = cms.InputTag("l1tCaloLayer1SpyDigis")
process.layer1Validator.emulSource = cms.InputTag("simCaloStage2Layer1Digis")
process.layer1Validator.verbose = cms.bool(True)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(162) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/dasu/l1tCaloLayer1Spy.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/data/dasu/l1tCaloLayer1Spy+Emulator.root'),
    outputCommands = cms.untracked.vstring('keep *')
)

process.p = cms.Path(process.simCaloStage2Layer1Digis+process.layer1Validator)

process.e = cms.EndPath(process.out)
