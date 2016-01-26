import FWCore.ParameterSet.Config as cms

process = cms.Process("L1TCaloLayer1Test")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('EventFilter.L1TCaloLayer1RawToDigi.l1tCaloLayer1Digis_cfi')
process.load('L1Trigger.L1TCaloLayer1.layer1EmulatorDigis_cfi')
process.layer1EmulatorDigis.useLUT = cms.bool(True)
process.layer1EmulatorDigis.verbose = cms.bool(False)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/dasu/Sept25_905539F2-5363-E511-831A-02163E013687.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/data/dasu/l1tCaloLayer1.root'),
    outputCommands = cms.untracked.vstring('keep *')
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_condDBv2_cff')
process.GlobalTag.globaltag = '74X_dataRun2_Express_v1'

process.p = cms.Path(process.l1tCaloLayer1Digis*process.layer1EmulatorDigis)

process.e = cms.EndPath(process.out)
