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

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#process.load("L1Trigger.Configuration.L1Config_cff")
#process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
#process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_Unprescaled_cff")

#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.load("DQMOffline.Trigger.L1TMonitor_dqmoffline_cff")
#process.load("DQMOffline.Trigger.Tau.HLTTauDQMOffline_cff")
#process.load("DQMOffline.Trigger.EgammaHLTOffline_cfi")
#process.load("Geometry.CaloEventSetup.CaloTopology_cfi")




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames = 
	cms.untracked.vstring('file:test.root')
        #cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/0C3B40D7-F87D-DD11-A9FB-000423D998BA.root','/store/relval/CMSSW_2_1_7/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/3A5455F3-F87D-DD11-AEF4-000423D94534.root')
#        cms.untracked.vstring(
#				'/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/06F01F20-E9DD-DD11-956F-001617E30CA4.root',
#				'/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/4CC9FEE5-EDDD-DD11-BA25-000423D99CEE.root'
#				)

                            
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

process.psource = cms.Path(process.hltResults)
process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


