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
process.load("DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi")
process.load("DQM.HLTEvF.FourVectorHLTOnline_cfi")
#process.load("DQMOffline.Trigger.L1TMonitor_dqmoffline_cff")
#process.load("DQMOffline.Trigger.Tau.HLTTauDQMOffline_cff")
#process.load("DQMOffline.Trigger.EgammaHLTOffline_cfi")
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

from DQM.HLTEvF.FourVectorHLTOnline_cfi import *
process.hltmonitoron = cms.Sequence(hltResultsOn)
hltResultsOn.triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
hltResultsOn.triggerResultsLabel = cms.InputTag("TriggerResults","","HLT")

from DQMOffline.Trigger.FourVectorHLTOffline_cfi import *
process.hltmonitor = cms.Sequence(hltResults)
hltResults.triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
hltResults.triggerResultsLabel = cms.InputTag("TriggerResults","","HLT")

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi import *
process.hltclient = cms.Sequence(hltFourVectorClient)
hltFourVectorClient.prescaleLS = cms.untracked.int32(-1)
hltFourVectorClient.monitorDir = cms.untracked.string('')
hltFourVectorClient.prescaleEvt = cms.untracked.int32(1)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
	#cms.untracked.vstring('file:test.root')
        cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/0C3B40D7-F87D-DD11-A9FB-000423D998BA.root','/store/relval/CMSSW_2_1_7/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/3A5455F3-F87D-DD11-AEF4-000423D94534.root')
                            
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

process.psource = cms.Path(process.hltmonitoron*process.hltmonitor*process.hltclient)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


process.DQMStore.referenceFileName = 'DQM_V0001_HLTOffline_R000000001_RelValZeeZmmMinBias.root'
