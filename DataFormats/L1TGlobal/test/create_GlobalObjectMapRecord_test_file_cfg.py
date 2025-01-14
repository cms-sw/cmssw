import FWCore.ParameterSet.Config as cms
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test GlobalObjectMapRecord')

parser.add_argument('--outputFileName', type=str, help='Output file name (default: testGlobalObjectMapRecord.root)', default='testGlobalObjectMapRecord.root')
args = parser.parse_args()

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.globalObjectMapRecordProducer = cms.EDProducer("TestWriteGlobalObjectMapRecord",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    nGlobalObjectMaps = cms.uint32(2),
    algoNames = cms.vstring('muonAlgo', 'electronAlgo'),
    algoBitNumbers = cms.vint32(11, 21),
    algoResults = cms.vint32(1, 0),
    tokenNames0     = cms.vstring('testnameA', 'testNameB'),
    tokenNumbers0 = cms.vint32(101, 102),
    tokenResults0 = cms.vint32(1, 0),
    tokenNames3 = cms.vstring('testNameC', 'testNameD'),
    tokenNumbers3 = cms.vint32(1001, 1002),
    tokenResults3 = cms.vint32(0, 1),
    nElements1 = cms.uint32(3),
    nElements2 = cms.uint32(4),
    nElements3 = cms.uint32(5),
    firstElement = cms.int32(11),
    elementDelta = cms.int32(3),
    bxIndexModulus = cms.uint32(3),
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(f'{args.outputFileName}')
)

process.path = cms.Path(process.globalObjectMapRecordProducer)
process.endPath = cms.EndPath(process.out)
