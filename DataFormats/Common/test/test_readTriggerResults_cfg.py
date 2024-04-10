import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))
process.maxEvents.input = 1

process.testReadTriggerResults = cms.EDAnalyzer("TestReadTriggerResults",
    triggerResultsTag = cms.InputTag("triggerResultsProducer", "", "PROD"),
        expectedParameterSetID = cms.string('8b99d66b6c3865c75e460791f721202d'),
        expectedNames = cms.vstring(),
        expectedHLTStates = cms.vuint32(0, 1, 2, 3),
        expectedModuleIndexes = cms.vuint32(11, 21, 31, 41)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTriggerResults2.root')
)

process.path = cms.Path(process.testReadTriggerResults)

process.endPath = cms.EndPath(process.out)
