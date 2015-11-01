import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

# DQM service
process.load("DQMServices.Core.DQMStore_cfi")

# Global Tag
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(globaltag = 'auto:run2_hlt_GRun')
process.GlobalTag.connect   = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

# Source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/data/Run2015D/L1Accept/RAW/v1/000/259/721/00000/52AFE8AC-6D78-E511-AF38-02163E0136CA.root'
  )
)

process.load("DQM.HLTEvF.triggerBxMonitor_cfi")
process.triggerBxMonitor.l1tResults = cms.untracked.InputTag('hltGtDigis', '', 'HLT')
process.triggerBxMonitor.hltResults = cms.untracked.InputTag('TriggerResults', '', 'HLT')

process.load('DQMServices.Components.DQMFileSaver_cfi')
process.dqmSaver.workflow = "/HLT/TriggerBxMonitor/All"

process.endp = cms.EndPath( process.triggerBxMonitor + process.dqmSaver )
