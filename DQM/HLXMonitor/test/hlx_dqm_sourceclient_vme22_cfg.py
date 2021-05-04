import FWCore.ParameterSet.Config as cms

process = cms.Process("hlxdqmlive")
process.load("DQMServices.Components.MessageLogger_cfi")

process.load("DQM.HLXMonitor.hlx_dqm_sourceclient_vme22_cfi")

## For private server vme22 use an empty source
process.source = cms.Source("EmptySource")

## For testing dqmEnv ... for online
##process.load("DQM.Integration.test.inputsource_cfi")
##process.EventStreamHttpReader.consumerName = 'HLX DQM Consumer'
##process.EventStreamHttpReader.sourceURL = cms.string('http://srv-c2d05-05:50082/urn:xdaq-application:lid=29')

process.load("DQM.Integration.test.environment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.hlxQualityTester = DQMQualityTester(
    # default is 1
    prescaleFactor = cms.untracked.int32(10000),
    # use eventloop for testing only ! default is false
    # untracked bool testInEventloop = false
    # use qtest on endRun or endJob
    # untracked bool qtestOnEndRun = false
    # untracked bool qtestOnEndJob = false
    qtList = cms.untracked.FileInPath('DQM/HLXMonitor/test/HLXQualityTests.xml')
)

##process.p = cms.Path(process.hlxdqmsource*process.hlxQualityTester*process.dqmSaver)
##process.p = cms.Path(process.hlxdqmsource*process.hlxQualityTester*process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.hlxdqmsource*process.hlxQualityTester)
process.hlxdqmsource.outputDir = '/opt/dqm/data/live'
process.hlxdqmsource.PrimaryHLXDAQIP = 'vmepcs2f17-22'
process.hlxdqmsource.SecondaryHLXDAQIP = 'vmepcs2f17-19'
process.hlxdqmsource.SourcePort = 51007
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9190
process.DQMStore.verbose = 0
process.dqmEnv.subSystemFolder = 'HLX'
process.dqmSaver.dirName = '/opt/dqm/data/tmp'
##process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.saveByTime = 4
process.dqmSaver.saveByMinute = 8


