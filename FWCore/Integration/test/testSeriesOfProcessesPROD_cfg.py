
# This configuration is designed to be run as the second
# in a series of cmsRun processes.  The process it configures
# will read a file in streamer format and produces two root
# files.

# For later event selection tests these paths are run:
#   path p1 1:25 pass
#   path p2 pass 51:60

# Checks the path names returned by the TriggerNames
# service.

# Multiple products are put in the event for use
# in subsequent processes.

# Two output files are created, one contains some
# fake raw data, the other contains some fake
# HLTDebug data (actual just dummy products containing
# an int, just for test purposes)

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring('file:testSeriesOfProcessesHLT.dat')
)

process.f1 = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(25),
  onlyOne = cms.untracked.bool(False)
)

process.f2a = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(50),
  onlyOne = cms.untracked.bool(False)
)

process.f2b = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(10),
  onlyOne = cms.untracked.bool(False)
)

process.a = cms.EDAnalyzer(
  "TestTriggerNames",
  trigPathsPrevious = cms.untracked.vstring(
    'p01', 
    'p02', 
    'p03', 
    'p04'
  ),
  streamerSource = cms.untracked.bool(True),
  trigPaths = cms.untracked.vstring('p1', 'p2'),
  dumpPSetRegistry = cms.untracked.bool(False)
)

# This puts products in the lumi's and run's.  One failure
# mode of the maxLuminosityBlock parameter is tested by their
# mere existence.
process.makeRunLumiProducts = cms.EDProducer("ThingWithMergeProducer")

# In the next process we want to test input from a secondary input
# file so we split the products over 2 output files.

process.out1 = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testSeriesOfProcessesPROD1.root'),
  outputCommands = cms.untracked.vstring(
    "drop *",
    "keep *_fakeRaw_*_*"
  )
)

process.out2 = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testSeriesOfProcessesPROD2.root'),
  outputCommands = cms.untracked.vstring(
    "keep *",
    "drop *_fakeRaw_*_*"
  )
)

process.pathanalysis = cms.EDAnalyzer("PathAnalyzer")

process.p1 = cms.Path(process.a * process.f1 * process.makeRunLumiProducts)
process.p2 = cms.Path(~process.f2a * process.f2b)

process.e = cms.EndPath(process.pathanalysis * process.out1 * process.out2)
