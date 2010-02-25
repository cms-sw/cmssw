import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'DT DQM Consumer'
#process.EventStreamHttpReader.sourceURL = cms.string('http://srv-c2d04-30:50082/urn:xdaq-application:lid=29')
process.EventStreamHttpReader.sourceURL = cms.string('http://srv-c2c05-07:22100/urn:xdaq-application:lid=30')

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQMServices.Core.DQM_cfg")
#process.DQMStore.referenceFileName = "DT_reference.root"

process.load("DQMServices.Components.DQMEnvironment_cfi")


#----------------------------
#### DQM Live Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9190
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '/localdatadisk/DTDQM/dqmdata'
process.dqmSaver.producer = 'DQM'

process.dqmSaver.saveByTime = -1
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByMinute = -1
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
#-----------------------------


# DT reco and DQM sequences
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")
#---- for P5 (online) DB access
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

# process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
# process.GlobalTag.globaltag = "CRAFT_V4H::All"
#---- for offline DB
#process.GlobalTag.globaltag = "CRAFT_V2P::All"

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

process.dtDQMPathPhys = cms.Path(process.unpackers + process.dqmmodules + process.physicsEventsFilter * process.dtDQMPhysSequence)

process.dtDQMPathCalib = cms.Path(process.unpackers + process.dqmmodules + process.calibrationEventsFilter * process.dtDQMCalib)

