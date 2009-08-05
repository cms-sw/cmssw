import FWCore.ParameterSet.Config as cms

process = cms.Process("hlxdqmlive")
process.load("DQMServices.Components.MessageLogger_cfi")

process.load("DQM.HLXMonitor.hlx_dqm_sourceclient_vme22_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.hlxQualityTester = cms.EDFilter("QualityTester",
    # default is 1
    prescaleFactor = cms.untracked.int32(10000),
    # use eventloop for testing only ! default is false
    # untracked bool testInEventloop = false
    # use qtest on endRun or endJob
    # untracked bool qtestOnEndRun = false
    # untracked bool qtestOnEndJob = false
    qtList = cms.untracked.FileInPath('DQM/HLXMonitor/test/HLXQualityTests.xml')
)

process.p = cms.Path(process.dqmEnv+process.hlxdqmsource+process.hlxQualityTester+process.dqmSaver)
process.hlxdqmsource.outputDir = '/opt/dqm/data/live'
process.hlxdqmsource.PrimaryHLXDAQIP = 'vmepcs2f17-18'
process.hlxdqmsource.SecondaryHLXDAQIP = 'vmepcs2f17-19'
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9190
process.DQMStore.verbose = 0
process.dqmEnv.subSystemFolder = 'HLX'
process.dqmSaver.dirName = '/opt/dqm/data/test'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = False

