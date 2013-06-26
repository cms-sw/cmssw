
# Make sure we can read and write the end result of
# all the merge and secondary file input testing

# Also test the event duplicate checking code and the
# skip event code in PoolSource

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
    , duplicateCheckMode = cms.untracked.string('checkEachFile')
    , skipEvents = cms.untracked.uint32(3)
    , noEventSort = cms.untracked.bool(False)
)

process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
1, 0, 0,
1, 1, 0,
1, 1, 4,
1, 1, 5,
1, 1, 6,
1, 1, 7,
1, 1, 8,
1, 1, 9,
1, 1, 10,
1, 1, 11,
1, 1, 12,
1, 1, 13,
1, 1, 14,
1, 1, 15,
1, 1, 16,
1, 1, 17,
1, 1, 18,
1, 1, 19,
1, 1, 20,
1, 1, 21,
1, 1, 22,
1, 1, 23,
1, 1, 24,
1, 1, 25,
1, 1, 0,
1, 0, 0,
2, 0, 0,
2, 1, 0,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 0,
2, 0, 0,
11, 0, 0,
11, 1, 0,
11, 1, 1,
11, 1, 0,
11, 2, 0,
11, 2, 1,
11, 2, 0,
11, 0, 0,
100, 0, 0,
100, 100, 0,
100, 100, 100,
100, 100, 0,
100, 0, 0,
1, 0, 0,
1, 1, 0,
1, 1, 21,
1, 1, 22,
1, 1, 23,
1, 1, 24,
1, 1, 25,
1, 1, 0,
1, 0, 0,
2, 0, 0,
2, 1, 0,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 0,
2, 0, 0,
1, 0, 0,
1, 1, 0,
1, 1, 1,
1, 1, 2,
1, 1, 3,
1, 1, 4,
1, 1, 5,
1, 1, 6,
1, 1, 7,
1, 1, 8,
1, 1, 9,
1, 1, 10,
1, 1, 0,
1, 0, 0
)
)
# At this point the first input file is done and
# we start with the second input file here.
process.test.expectedRunLumiEvents.extend([
1, 0, 0,
1, 1, 0,
1, 1, 1,
1, 1, 2,
1, 1, 3,
1, 1, 4,
1, 1, 5,
1, 1, 6,
1, 1, 7,
1, 1, 8,
1, 1, 9,
1, 1, 10,
1, 1, 11,
1, 1, 12,
1, 1, 13,
1, 1, 14,
1, 1, 15,
1, 1, 16,
1, 1, 17,
1, 1, 18,
1, 1, 19,
1, 1, 20,
1, 1, 21,
1, 1, 22,
1, 1, 23,
1, 1, 24,
1, 1, 25,
1, 1, 0,
1, 0, 0,
2, 0, 0,
2, 1, 0,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 0,
2, 0, 0,
11, 0, 0,
11, 1, 0,
11, 1, 1,
11, 1, 0,
11, 2, 0,
11, 2, 1,
11, 2, 0,
11, 0, 0,
100, 0, 0,
100, 100, 0,
100, 100, 100,
100, 100, 0,
100, 0, 0,
1, 0, 0,
1, 1, 0,
1, 1, 21,
1, 1, 22,
1, 1, 23,
1, 1, 24,
1, 1, 25,
1, 1, 0,
1, 0, 0,
2, 0, 0,
2, 1, 0,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 0,
2, 0, 0,
1, 0, 0,
1, 1, 0,
1, 1, 1,
1, 1, 2,
1, 1, 3,
1, 1, 4,
1, 1, 5,
1, 1, 6,
1, 1, 7,
1, 1, 8,
1, 1, 9,
1, 1, 10,
1, 1, 0,
1, 0, 0
])

process.path1 = cms.Path(process.test)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:testRunMergeRecombinedCopied1.root')
)

process.endpath1 = cms.EndPath(process.out)
