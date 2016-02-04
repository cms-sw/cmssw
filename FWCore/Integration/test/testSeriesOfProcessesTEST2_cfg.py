
# Test the case of a duplicate process name being used.
# This should fail with an exception error message that
# indicates a duplicate process name was used.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSeriesOfProcessesTEST.root'),
)

# Add one module in a path to force the process to be
# added to the process history.  This could be any module.
# The module serves no other purpose.
process.a = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag()
)

process.test1 = cms.Path(process.a)
