import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))
process.maxEvents.input = 1

process.testReadRawDataBuffer = cms.EDAnalyzer("TestReadRawDataBuffer",
    rawDataBufferTag = cms.InputTag("rawDataBufferProducer", "", "PROD"),
    dataPattern1 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    dataPattern2 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRawDataBuffer2.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.testReadRawDataBuffer)

process.endPath = cms.EndPath(process.out)
