import FWCore.ParameterSet.Config as cms

process = cms.Process("hlxdqmlive")

from FWCore.MessageLogger.MessageLogger_cfi import *

## Input source
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'HLX DQM Consumer'
process.EventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputHLTDQM')

## HLX configuration
process.load("DQM.HLXMonitor.hlx_dqm_sourceclient_cfi")
process.hlxdqmsource.PrimaryHLXDAQIP = 'vmepcs2f17-22'
process.hlxdqmsource.SecondaryHLXDAQIP = 'vmepcs2f17-18'
process.hlxdqmsource.SourcePort = 51007

## Set up env and saver
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder    = "HLX"

## Lumi reference file
process.DQMStore.referenceFileName = '/dqmdata/dqm/reference/hlx_reference.root'

process.hlxQualityTester = cms.EDAnalyzer("QualityTester",
    # default is 1
    prescaleFactor = cms.untracked.int32(10000),
    # use eventloop for testing only ! default is false
    # untracked bool testInEventloop = false
    qtList = cms.untracked.FileInPath('DQM/HLXMonitor/test/HLXQualityTests.xml'),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True)
)

process.p = cms.Path(process.hlxdqmsource*process.hlxQualityTester*process.dqmEnv*process.dqmSaver)

## Shouldn't need this anymore ...
##process.hlxdqmsource.outputDir = process.dqmSaver.dirName


