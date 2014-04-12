import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalTimingTest")

process.load("DQM.Integration.test.inputsource_playback_cfi")
process.EventStreamHttpReader.consumerName = 'Hcal Timing DQM Consumer'

process.MessageLogger = cms.Service("MessageLogger",
     categories   = cms.untracked.vstring(''),
     destinations = cms.untracked.vstring('cout'),
     debugModules = cms.untracked.vstring('*'),
     cout = cms.untracked.PSet(
         threshold = cms.untracked.string('WARNING'),
         WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0))
     )
)

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff")
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = 'source'

process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("L1Trigger.L1ExtraFromDigis.l1extraParticles_cff")
process.load("HLTrigger.HLTfilters.hltLevel1GTSeed_cfi")

process.load("DQM.HcalMonitorModule.HcalTimingModule_cfi")

#-----------------------------
# Hcal Conditions: from Global Conditions Tag 
#-----------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = 'CRAFT_V4H::All' # or any other appropriate
process.prefer("GlobalTag")


### include to get DQM histogramming services
process.load("DQMServices.Core.DQM_cfg")

### set the verbose
process.DQMStore.verbose = 0


#### BEGIN DQM Online Environment #######################

### replace YourSubsystemName by the name of your source ###
### use it for dqmEnv, dqmSaver
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQM.Integration.test.environment_playback_cfi")
### path where to save the output file
#process.dqmSaver.dirName = '.'
### the filename prefix
#process.dqmSaver.producer = 'DQM'
### possible conventions are "Online", "Offline" and "RelVal"
#process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'HcalTimingMonitor'

process.p = cms.Path(process.hcalDigis*process.l1GtUnpack*process.hcalTimingMonitor*process.dqmEnv*process.dqmSaver)


