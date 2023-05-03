import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[2]))
process.maxEvents.input = 1

process.testReadFEDRawDataCollection = cms.EDAnalyzer("TestReadFEDRawDataCollection",
    fedRawDataCollectionTag = cms.InputTag("fedRawDataCollectionProducer", "", "PROD"),
    expectedFEDData0 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7),
    expectedFEDData3 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testFEDRawDataCollection2.root')
)

process.path = cms.Path(process.testReadFEDRawDataCollection)

process.endPath = cms.EndPath(process.out)
