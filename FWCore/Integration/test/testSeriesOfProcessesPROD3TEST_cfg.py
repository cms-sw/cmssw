
# This configuration is designed to be run as the last
# in a series of cmsRun processes.

# Tests the maxLuminosityBlocks parameter

# checks to see that the process level fakeRaw overrides the file based one

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD3TEST")

process.maxLuminosityBlocks = cms.untracked.PSet(
  input = cms.untracked.int32(3)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSeriesOfProcessesPROD2TEST.root'),
)

# This module tests to see if the products put in at the second step
# (the fake PROD step) survived through to the last file.  At the PROD
# stage the products were split into two files so this test secondary
# file input.
process.a = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("fakeRaw"),
                                       cms.InputTag("fakeHLTDebug") ),

  # Test the maxLuminosityBlock parameter
  # 3 luminosity blocks contain 15 events
  # Each event contains one product with a value of 20 and
  # one product with a value of 1000
  # If the maxLuminosityBlock parameter is working correctly the
  # following should be the sum of all the values.
  # The product values are hard coded into the fake
  # HLT configuration (the first one in this series).
  expectedSum = cms.untracked.int32(15300)
)

process.test1 = cms.Path(process.a)
