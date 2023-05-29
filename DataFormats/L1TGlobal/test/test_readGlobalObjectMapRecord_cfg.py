import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[2]))
process.maxEvents.input = 1

process.testReadGlobalObjectMapRecord = cms.EDAnalyzer("TestReadGlobalObjectMapRecord",
    expectedAlgoNames = cms.vstring('muonAlgo', 'electronAlgo'),
    expectedAlgoBitNumbers = cms.vint32(11, 21),
    expectedAlgoGtlResults = cms.vint32(1, 0),
    expectedTokenNames0 = cms.vstring('testnameA', 'testNameB'),
    expectedTokenNumbers0 = cms.vint32(101, 102),
    expectedTokenResults0 = cms.vint32(1, 0),
    expectedTokenNames3 = cms.vstring('testNameC', 'testNameD'),
    expectedTokenNumbers3 = cms.vint32(1001, 1002),
    expectedTokenResults3 = cms.vint32(0, 1),
    expectedFirstElement = cms.int32(11),
    expectedElementDelta = cms.int32(3),
    # 3 (delta) * (3*4*5 + 3*4) + 11 = 227
    expectedFinalValue = cms.int32(227),
    globalObjectMapRecordTag = cms.InputTag("globalObjectMapRecordProducer", "", "PROD"),
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGlobalObjectMapRecord2.root')
)

process.path = cms.Path(process.testReadGlobalObjectMapRecord)

process.endPath = cms.EndPath(process.out)
