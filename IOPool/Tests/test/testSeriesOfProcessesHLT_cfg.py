
# This configuration is designed to be run as the first
# in a series of cmsRun processes.  Several things get
# tested independently in this series of processes.
# This first process will create a streamer file.

# For event selection tests several paths are run:
#   99 events are generated
#   path p01 events 1:98 pass
#   path p02 events 21:30 pass
#   path p03 event 71 only passes
#   path p04 all fail

# Checks the path names returned by the TriggerNames
# service.

# Multiple products are put in the event for use
# in subsequent processes. Some products faking
# raw data products and some faking hlt products.
# They are not used here, just created for later
# use.

# Creates multiple luminosity blocks for a later
# test of the maxLuminosityBlock parameter

import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

process.source = cms.Source("EmptySource",
  firstLuminosityBlock = cms.untracked.uint32(1),
  numberEventsInLuminosityBlock = cms.untracked.uint32(5),
  firstEvent = cms.untracked.uint32(1),
  firstRun = cms.untracked.uint32(1),
  numberEventsInRun = cms.untracked.uint32(1000)
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(99)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

#import FWCore.Framework.test.cmsExceptionsFatal_cff
#process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.f1 = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(98),
  onlyOne = cms.untracked.bool(False)
)

process.f2a = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(20),
  onlyOne = cms.untracked.bool(False)
)

process.f2b = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(10),
  onlyOne = cms.untracked.bool(False)
)

process.f3 = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(71),
  onlyOne = cms.untracked.bool(True)
)

process.f4 = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(101),
  onlyOne = cms.untracked.bool(True)
)

process.a = cms.EDAnalyzer(
  "TestTriggerNames",
  trigPaths = cms.untracked.vstring(
    'p01', 
    'p02', 
    'p03', 
    'p04'
  ),
  endPaths = cms.untracked.vstring('e'),
  dumpPSetRegistry = cms.untracked.bool(False)
)

process.fakeRaw = cms.EDProducer(
  "IntProducer",
  ivalue = cms.int32(10)
)

process.fakeHLTDebug = cms.EDProducer(
  "IntProducer",
  ivalue = cms.int32(1000)
)

process.out = cms.OutputModule("EventStreamFileWriter",
  fileName = cms.untracked.string('testSeriesOfProcessesHLT.dat'),
  compression_level = cms.untracked.int32(1),
  use_compression = cms.untracked.bool(True),
  max_event_size = cms.untracked.int32(7000000)
)

process.p01 = cms.Path(process.f1)
process.p02 = cms.Path(~process.f2a*process.f2b)
process.p03 = cms.Path(process.f3)
process.p04 = cms.Path(process.a *
                       process.fakeRaw * process.fakeHLTDebug *
                       process.f4)

process.e = cms.EndPath(process.out)
