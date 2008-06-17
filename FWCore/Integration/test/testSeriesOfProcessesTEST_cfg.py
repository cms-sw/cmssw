
# This configuration is designed to be run as the third
# in a series of cmsRun processes.  Several things get
# tested.

# For event selection tests
#   path1 even pass
#   path2 1:40 pass

# Checks the path names returned by the TriggerNames
# service.

# We read both files previously written and test that
# the secondary input file feature of the PoolSource
# works even in the case when the products went through
# a streamer file.

# The SewerModule OutputModule's test the SelectEvents
# feature.  If the expected number of events does not
# pass the selection, they abort with an error message.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSeriesOfProcessesPROD2.root'),
  secondaryFileNames = cms.untracked.vstring('file:testSeriesOfProcessesPROD1.root')
)

process.f1 = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(2),
  onlyOne = cms.untracked.bool(True)
)

process.f2 = cms.EDFilter("TestFilterModule",
  acceptValue = cms.untracked.int32(40),
  onlyOne = cms.untracked.bool(False)
)

process.a = cms.EDAnalyzer("TestTriggerNames",
  trigPathsPrevious = cms.untracked.vstring('p1', 'p2'),
  trigPaths = cms.untracked.vstring(
    'path1', 
    'path2', 
    'path3', 
    'path4', 
    'path5', 
    'path6', 
    'path7', 
    'path8'),
  dumpPSetRegistry = cms.untracked.bool(False)
)

process.out1 = cms.OutputModule("SewerModule",
  shouldPass = cms.int32(60),
  name = cms.string('out1'),
  SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('p02:HLT', 
      ' p03    :       HLT', 
      'p2:PROD', 
      'path1:TEST')
  )
)

process.out2 = cms.OutputModule("SewerModule",
  shouldPass = cms.int32(98),
  name = cms.string('out2'),
  SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('*:HLT')
  )
)

process.out3 = cms.OutputModule("SewerModule",
  shouldPass = cms.int32(64),
  name = cms.string('out3'),
  SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('!*:PROD')
  )
)

process.out4 = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testSeriesOfProcessesTEST.root'),
)

process.path1 = cms.Path(process.f1)
process.path2 = cms.Path(process.f2)
process.path3 = cms.Path(process.f1)
process.path4 = cms.Path(process.f2)
process.path5 = cms.Path(process.f1)
process.path6 = cms.Path(process.f2)
process.path7 = cms.Path(process.f1)
process.path8 = cms.Path(process.f2*process.a)

process.e = cms.EndPath(process.out1+process.out2+process.out3+process.out4)
