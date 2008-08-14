import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.CSCMonitorModule.test.csc_dqm_sourceclient_cfi")

process.load("DQMServices.Components.test.dqm_onlineEnv_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('file:/tmp/valdo/GlobalCruzet1.00043553.0227.A.storageManager.3.0000.dat')
)

process.p = cms.Path(process.dqmClient+process.dqmEnv+process.dqmSaver)
process.DQMStore.referenceFileName = '/home/dqmdevlocal/reference/csc_reference.root'
process.DQM.collectorHost = 'srv-c2d05-16'
process.DQM.collectorPort = 9090
process.dqmSaver.dirName = '.'
process.dqmEnv.subSystemFolder = 'CSC'
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


