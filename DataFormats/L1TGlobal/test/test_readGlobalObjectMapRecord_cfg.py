import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test GlobalObjectMapRecord')

parser.add_argument('--globalObjectMapClassVersion', type=int, help='Class version of GlobalObjectMap (default: 10)', default=10)
parser.add_argument('--inputFileName', type=str, help='Input file name (default: testGlobalObjectMapRecord.root)', default='testGlobalObjectMapRecord.root')
parser.add_argument('--outputFileName', type=str, help='Output file name (default: testGlobalObjectMapRecord2.root)', default='testGlobalObjectMapRecord2.root')
args = parser.parse_args()

process = cms.Process('READ')

process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(f'file:{args.inputFileName}'))
process.maxEvents.input = 1

process.testReadGlobalObjectMapRecord = cms.EDAnalyzer('TestReadGlobalObjectMapRecord',
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
    expectedBxIndexModulus = cms.uint32(3),
    # 3 (delta) * (3*4*5 + 3*4) + 11 = 227
    expectedFinalValue = cms.int32(227),
    globalObjectMapRecordTag = cms.InputTag('globalObjectMapRecordProducer', '', 'PROD'),
    globalObjectMapClassVersion = cms.uint32(args.globalObjectMapClassVersion),
)

process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string(f'{args.outputFileName}'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.testReadGlobalObjectMapRecord)

process.endPath = cms.EndPath(process.out)
