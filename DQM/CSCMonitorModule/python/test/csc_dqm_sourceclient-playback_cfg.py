import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQM")

process.load("DQM.Integration.test.inputsource_playback_cfi")
process.load("DQM.CSCMonitorModule.test.csc_dqm_sourceclient_cfi")
process.load("DQM.Integration.test.environment_playback_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.source = cms.Source("EventStreamHttpReader",
#    sourceURL = cms.string('http://localhost:50082/urn:xdaq-application:lid=29'),
#    consumerPriority = cms.untracked.string('normal'),
#    max_event_size = cms.int32(7000000),
#    consumerName = cms.untracked.string('CSC DQM Source'),
#    max_queue_depth = cms.int32(5),
#    maxEventRequestRate = cms.untracked.double(10.0),
#    SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('*')
#    ),
#    headerRetryInterval = cms.untracked.int32(3)
#)

process.p = cms.Path(process.dqmClient+process.dqmEnv+process.dqmSaver)

#process.DQMStore.referenceFileName = '/home/dqmdevlocal/reference/csc_reference.root'
process.DQMStore.referenceFileName = '/afs/cern.ch/user/v/valdo/CMSSW_2_1_0/src/DQM/CSCMonitorModule/data/csc_reference.root'

process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
process.DQM.collectorPort = 9090

process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'

process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.dqmEnv.subSystemFolder = 'CSC'
