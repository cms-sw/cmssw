import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQM")

#----------------------------
#### Event Source
#----------------------------
process.load("DQM.DTMonitorModule.test.NewEventStreamFileReader_cfi")
#process.EventStreamHttpReader.consumerName = 'DT DQM Consumer'


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
process.DQM.collectorHost = 'pippo'
process.DQM.collectorPort = 9199
process.dqmEnv.subSystemFolder = 'DT'
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
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
process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V4H::All"
#---- for offline DB
#process.GlobalTag.globaltag = "CRAFT_V2P::All"

# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

process.dtDQMPathPhys = cms.Path(process.unpackers + process.dqmmodules + process.reco + process.dtDQMTask + process.dtDQMTest)

process.dtDQMPathCalib = cms.Path(process.unpackers + process.dqmmodules + process.calibrationEventsFilter * process.dtDQMCalib)

