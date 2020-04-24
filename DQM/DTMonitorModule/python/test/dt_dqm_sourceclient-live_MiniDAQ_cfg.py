
import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.DTMonitorModule.test.inputsource_MiniDAQ_cfi")

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")
#process.DQMStore.referenceFileName = "/dtdata/dqmdata/global/dt_reference.root"

process.load("DQMServices.Components.DQMEnvironment_cfi")


#----------------------------
#### DQM Live Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9991
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '/dtdata/dqmdata/minidaq'
process.dqmSaver.producer = 'DQM'

process.dqmSaver.saveByTime = -1
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByMinute = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
#-----------------------------


# DT reco and DQM sequences
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")

#---- for P5 (online) DB access
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")


# ------------------------------------------------
# Special settings for commissioning RUNS
process.dtTPmonitor.defaultTtrig = 3250
process.dtTPmonitor.defaultTmax = 300
process.dtDigiMonitor.readDB = False
process.dtDigiMonitor.filterSyncNoise = False
process.dtDigiMonitor.lookForSyncNoise = True
process.dtDigiMonitor.checkNoisyChannels = False
# ------------------------------------------------

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('DTSynchNoise'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              INFO = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              DTSynchNoise = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(-1))
                                                              )
                                    )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

process.dtDQMPathPhys = cms.Path(process.unpackers + process.dqmmodules * process.reco + process.dtDQMTask + process.dtDQMTest)

process.dtDQMPathCalib = cms.Path(process.unpackers + process.dqmmodules + process.calibrationEventsFilter *  process.dtDQMCalib)

