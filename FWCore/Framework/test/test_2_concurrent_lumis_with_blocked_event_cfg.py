import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

from FWCore.Modules.modules import EmptySource
process.source = EmptySource(numberEventsInLuminosityBlock=1, numberEventsInRun=3)

process.maxEvents.input = 4

from FWCore.Framework.modules import edmtest_ConcurrentLumiAnalyzer
process.tester = edmtest_ConcurrentLumiAnalyzer()

process.p = cms.Path(process.tester)

process.options.numberOfThreads = 2
process.options.numberOfConcurrentLuminosityBlocks = 2

#process.add_(cms.Service("Tracer"))
