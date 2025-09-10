# This is intended to test whether fast cloning is
# actually occurring when it should. It uses a trick
# to check.  With this configuration, fast cloning should
# occur.  An exception unrelated to fast cloning is forced
# on the second event, and after that there should be a
# secondary exception as the Framework attempts to write out
# and close the root files.  This secondary exception occurs
# because the Event TTree should be imbalanced.  This imbalance
# is caused by fast cloning.  The TTree would not be imbalanced
# if fast cloning was not occurring.  The shell script that
# invokes cmsRun with this configuration then uses grep to
# look for the expected inbalance error and this verifies
# that fast cloning was occurring.

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge6.root'
    )
    , duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root')
    #, fastCloning = cms.untracked.bool(False)
)

process.testThrow = cms.EDAnalyzer("TestFailuresAnalyzer",
    whichFailure = cms.int32(5),
    eventToThrow = cms.untracked.uint64(2)
)

process.p = cms.Path(process.testThrow)

process.e = cms.EndPath(process.out)
