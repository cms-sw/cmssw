import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

process = cms.Process("CSCTPEmulator2", Run2_2018)
process.load("Configuration/StandardSequences/GeometryRecoDB_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load("DQM.L1TMonitorClient.L1TdeCSCTPGClient_cfi")
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')

process.maxEvents = cms.untracked.PSet(
      input = cms.untracked.int32(1)
)

process.options = cms.untracked.PSet(
      SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.source = cms.Source(
    "DQMRootSource",
    fileNames = cms.untracked.vstring("file:step_DQM.root")
)

process.MessageLogger = cms.Service("MessageLogger")

## global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

## schedule and path definition
process.dqm_step = cms.Path(process.l1tdeCSCTPGClient)
process.dqmsave_step = cms.Path(process.DQMSaver)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.schedule = cms.Schedule(
      process.dqm_step,
      process.endjob_step,
      process.dqmsave_step
)
