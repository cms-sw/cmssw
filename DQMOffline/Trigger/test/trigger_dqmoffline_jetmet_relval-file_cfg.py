
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

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQMOffline.Trigger.JetMETHLTOfflineSource_cfi")
process.load("DQMOffline.Trigger.JetMETHLTOfflineClient_cfi")
process.load("DQMOffline.Trigger.HLTJetMETQualityTester_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
 
# configure HLT
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('40 OR 41')

from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *

process.load("DQMServices.Components.DQMStoreStats_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring('/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V2-v1/0004/102CC3BD-642F-DF11-811A-003048678F26.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
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



#process.psource = cms.Path(process.hltLevel1GTSeed*process.jetMETHLTOfflineSource*process.hltJetMETOfflineQualityTests*process.dqmStoreStats) 
process.psource = cms.Path(process.hltLevel1GTSeed*process.jetMETHLTOfflineSource*process.jetMETHLTOfflineClient*process.hltJetMETOfflineQualityTests*process.dqmStoreStats)

process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


