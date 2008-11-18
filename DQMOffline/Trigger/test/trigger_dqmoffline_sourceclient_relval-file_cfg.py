import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#process.load("Configuration.GlobalRuns.ReconstructionGR_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("L1Trigger.Configuration.L1Config_cff")

#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQMOffline.Trigger.FourVectorHLTOffline_cfi")
#process.load("DQMOffline.Trigger.L1TMonitor_dqmoffline_cff")
#process.load("DQMOffline.Trigger.Tau.HLTTauDQMOffline_cff")
#process.load("DQMOffline.Trigger.EgammaHLTOffline_cfi")
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

from DQMOffline.Trigger.FourVectorHLTOffline_cfi import *
process.hltmonitor = cms.Sequence(hltResults)
hltResults.triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
hltResults.triggerResultsLabel = cms.InputTag("TriggerResults","","HLT")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring('file:test.root')
#cms.untracked.vstring('/store/relval/CMSSW_2_1_0/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/0EF324BD-9160-DD11-B591-000423D944F8.root')
#cms.untracked.vstring('/store/relval/CMSSW_2_1_0/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/08D532C8-9C60-DD11-AB1C-000423D99996.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)

process.psource = cms.Path(process.hltmonitor)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


