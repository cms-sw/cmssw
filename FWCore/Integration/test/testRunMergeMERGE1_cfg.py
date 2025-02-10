
# This configuration tests the lumisToSkip, firstRun,
# firstLuminosityBlock, and firstEvent parameters of
# the PoolSource.

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = [
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root',
        'file:testRunMerge4.root',
        'file:testRunMerge5.root'
    ]
    , firstRun = 17
    , firstLuminosityBlock = 3
    , firstEvent = 6
    , lumisToSkip = [
        '18:3',
        '19:2',
        '21:4',
        '16:2'
    ]
    , duplicateCheckMode = 'checkEachRealDataFile'
)

from FWCore.Integration.modules import ThingWithMergeProducer
process.thingWithMergeProducer = ThingWithMergeProducer()

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(fileName = 'testRunMerge_a.root')

from FWCore.Framework.modules import RunLumiEventAnalyzer
process.test = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
17, 0, 0,
17, 3, 0,
17, 3, 6,
17, 3, 0,
17, 4, 0,
17, 4, 7,
17, 4, 8,
17, 4, 9,
17, 4, 0,
17, 0, 0,
18, 0, 0,
18, 2, 0,
18, 2, 1,
18, 2, 2,
18, 2, 3,
18, 2, 0,
18, 4, 0,
18, 4, 7,
18, 4, 8,
18, 4, 9,
18, 4, 0,
18, 0, 0,
19, 0, 0,
19, 3, 0,
19, 3, 4,
19, 3, 5,
19, 3, 6,
19, 3, 0,
19, 4, 0,
19, 4, 7,
19, 4, 8,
19, 4, 9,
19, 4, 0,
19, 0, 0,
20, 0, 0,
20, 2, 0,
20, 2, 1,
20, 2, 2,
20, 2, 3,
20, 2, 0,
20, 3, 0,
20, 3, 4,
20, 3, 5,
20, 3, 6,
20, 3, 0,
20, 4, 0,
20, 4, 7,
20, 4, 8,
20, 4, 9,
20, 4, 0,
20, 0, 0,
21, 0, 0,
21, 2, 0,
21, 2, 1,
21, 2, 2,
21, 2, 3,
21, 2, 0,
21, 3, 0,
21, 3, 4,
21, 3, 5,
21, 3, 6,
21, 3, 0,
21, 0, 0
]
)

process.path1 = cms.Path(process.thingWithMergeProducer + process.test)
process.e = cms.EndPath(process.out)
