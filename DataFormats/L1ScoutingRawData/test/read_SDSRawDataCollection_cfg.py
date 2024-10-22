import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READRAW")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))
process.maxEvents.input = 1

process.testReadSDSDRawDataCollection = cms.EDAnalyzer("TestReadSDSRawDataCollection",
    sdsRawDataCollectionTag = cms.InputTag("sdsRawDataCollectionProducer", "", "PROD"),
    expectedSDSData1 = cms.vuint32(0, 1, 2, 3),
    expectedSDSData2 = cms.vuint32(42, 43, 44, 45)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSDSRawDataCollection2.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.testReadSDSDRawDataCollection)

process.endPath = cms.EndPath(process.out)
