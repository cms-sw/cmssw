import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.CSCMonitorModule.test.csc_dqm_sourceclient_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.source = cms.Source("EventStreamHttpReader",
    sourceURL = cms.string('http://srv-c2d05-14.cms:22100/urn:xdaq-application:lid=30'),
    consumerPriority = cms.untracked.string('normal'),
    max_event_size = cms.int32(7000000),
    consumerName = cms.untracked.string('CSC DQM Source'),
    max_queue_depth = cms.int32(5),
    maxEventRequestRate = cms.untracked.double(10.0),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*DQM')
    ),
    headerRetryInterval = cms.untracked.int32(3)
)

process.p = cms.Path(process.dqmClient+process.dqmEnv+process.dqmSaver)
process.DQMStore.referenceFileName = '/home/dqmdevlocal/reference/csc_reference.root'
process.DQM.collectorHost = 'srv-c2d05-19.cms'
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '/cms/mon/data/dropbox'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'CSC'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


