import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

# DQM service
process.load("DQMServices.Core.DQMStore_cfi")

# MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# Global Tag
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(globaltag = '80X_dataRun2_HLT_v12')
process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')

# Source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/data/Run2016B/HLTPhysics2/RAW/v1/000/272/022/00000/4CE23DEB-CB0D-E611-A6AC-02163E01181C.root'
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

# unpack L1 digis
process.load("EventFilter.L1TRawToDigi.gtStage2Digis_cfi")
process.gtStage2Digis.InputLabel = cms.InputTag( "rawDataCollector" )

process.load("DQM.HLTEvF.triggerBxMonitor_cfi")
process.triggerBxMonitor.l1tResults = cms.untracked.InputTag('gtStage2Digis')
process.triggerBxMonitor.hltResults = cms.untracked.InputTag('TriggerResults', '', 'HLT')

process.load('DQMServices.Components.DQMFileSaver_cfi')
process.dqmSaver.workflow = "/HLT/TriggerBxMonitor/All"

process.endp = cms.EndPath( process.gtStage2Digis + process.triggerBxMonitor + process.dqmSaver )
