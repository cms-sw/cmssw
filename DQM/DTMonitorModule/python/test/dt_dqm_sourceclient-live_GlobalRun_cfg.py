import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'DT Private Global DQM Testing Consumer'
#process.source.sourceURL = cms.string('http://dqm-c2d07-30:50082/urn:xdaq-application:lid=29')  # Playback Server
process.source.sourceURL = cms.string('http://dqm-c2d07-30:22100/urn:xdaq-application:lid=30')   # General use source for Private DQM
#process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_L1Mu*','HLT_L1DoubleMu*','HLT_Mu*','HLT_DoubleMu*','HLT_DTErrors'))

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.DQMStore.referenceFileName = "/dtdata/dqmdata/global/dt_reference.root"

process.load("DQMServices.Components.DQMEnvironment_cfi")


#----------------------------
#### DQM Live Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9991
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '/dtdata/dqmdata/global' 
#process.dqmSaver.dirName = '.' 
process.dqmSaver.producer = 'DQM'

process.dqmSaver.saveByTime = -1
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByMinute = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
#-----------------------------


# DT reco and DQM sequences
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
# ideal geometry for LUT task
process.load("Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff")
process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")
#---- for P5 (online) DB access
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

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

process.dtDQMPathPhys = cms.Path(process.unpackers + process.dqmmodules + process.physicsEventsFilter * process.dtDQMPhysSequence)

process.dtDQMPathCalib = cms.Path(process.unpackers + process.dqmmodules + process.calibrationEventsFilter * process.dtDQMCalib)

print process.source.sourceURL
