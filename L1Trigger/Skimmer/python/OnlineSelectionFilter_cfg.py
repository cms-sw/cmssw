import FWCore.ParameterSet.Config as cms

process = cms.Process("L1SKIM")
# initialize  MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("L1Trigger.Configuration.L1Emulator_cff")

process.load("L1Trigger.L1ExtraFromDigis.l1extraParticleMap_cfi")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(20),
    fileNames = cms.untracked.vstring('file:PhysVal-DiElectron-Ene10.root')
)

process.osFilter = cms.EDFilter("OnlineSelectionFilter",
    gtReadoutSource = cms.InputTag("l1extraParticleMap")
)

process.l1tOutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testL1TOutput.root'),
    outputCommands = cms.untracked.vstring('drop *', 'keep edmHepMCProduct_*_*_*', 'keep L1*_*_*_*', 'keep EcalTrigger*_*_*_*', 'keep HcalTrigger*_*_*_*', 'keep *_l1extraParticleMap_*_*', 'keep *_l1extraParticles_*_*')
)

process.hltOutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testHLTOutput.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('filterPath')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep edmHepMCProduct_*_*_*', 'keep *_g4SimHits_*_*', 'keep *_*_*_L1SKIM', 'drop Crossing*_*_*_*', 'drop *_ecalUnsuppressedDigis_*_*', 'drop EcalTrigger*_*_*_*', 'drop *CSCTriggerContainer_*_*_*', 'drop *_dttpgprod_*_*')
)

process.filterPath = cms.Path(process.l1emulator*process.l1extraParticleMap*process.osFilter)
process.l1t = cms.EndPath(process.l1tOutput)
process.hlt = cms.EndPath(process.hltOutput)
process.MessageLogger.cout.threshold = 'ERROR'
process.MessageLogger.cerr.default.limit = 10

