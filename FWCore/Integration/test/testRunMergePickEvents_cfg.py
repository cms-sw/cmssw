
# Test the PoolSource parameter that one can use
# to request specific events.  This interacts with
# the duplicate checking and skip event feature so
# I set those parameters also and test that.

import FWCore.ParameterSet.Config as cms

process = cms.Process("COPY")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode  = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeRecombined.root',
        'file:testRunMergeRecombined.root'
    )
    #, duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
    , skipEvents = cms.untracked.uint32(3)
    , eventsToProcess = cms.untracked.VEventRange('1:1-1:6')
)

process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
1, 0, 0,
1, 1, 0,
1, 1, 4,
1, 1, 5,
1, 1, 6,
1, 1, 0,
1, 0, 0,
)
)

process.path1 = cms.Path(process.test)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:testRunMergePickEvent.root')
)

process.endpath1 = cms.EndPath(process.out)
