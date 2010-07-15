
# This configuration tests the lumisToProcess and eventsToSkip
# parameters of the PoolSource.

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root',
        'file:testRunMerge4.root',
        'file:testRunMerge5.root'
    )
    , lumisToProcess = cms.untracked.VLuminosityBlockRange(
                                           '11:2',
                                           '15:2-15:8',
                                           '19:2-20:2',
                                           '21:4'
                                          )
    , eventsToSkip = cms.untracked.VEventRange(
                                           '19:6-19:8',
                                           '21:8'
                                           )
    , duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeMERGE3.root')
)

process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
1,  0, 0,
1,  0, 0,
2,  0, 0,
2,  0, 0,
1,  0, 0,
1,  0, 0,
11, 0, 0,
11, 2, 0,
11, 2, 1,
11, 2, 2,
11, 2, 3,
11, 2, 0,
11, 0, 0,
12, 0, 0,
12, 0, 0,
13, 0, 0,
13, 0, 0,
14, 0, 0,
14, 0, 0,
15, 0, 0,
15, 2, 0,
15, 2, 1,
15, 2, 2,
15, 2, 3,
15, 2, 0,
15, 3, 0,
15, 3, 4,
15, 3, 5,
15, 3, 6,
15, 3, 0,
15, 4, 0,
15, 4, 7,
15, 4, 8,
15, 4, 9,
15, 4, 0,
15, 0, 0,
16, 0, 0,
16, 0, 0,
17, 0, 0,
17, 0, 0,
18, 0, 0,
18, 0, 0,
19, 0, 0,
19, 2, 0,
19, 2, 1,
19, 2, 2,
19, 2, 3,
19, 2, 0,
19, 3, 0,
19, 3, 4,
19, 3, 5,
19, 3, 0,
19, 4, 0,
19, 4, 9,
19, 4, 0,
19, 0, 0,
20, 0, 0,
20, 2, 0,
20, 2, 1,
20, 2, 2,
20, 2, 3,
20, 2, 0,
20, 0, 0,
21, 0, 0,
21, 4, 0,
21, 4, 7,
21, 4, 9,
21, 4, 0,
21, 0, 0
)
)

process.path1 = cms.Path(process.thingWithMergeProducer + process.test)
process.e = cms.EndPath(process.out)
