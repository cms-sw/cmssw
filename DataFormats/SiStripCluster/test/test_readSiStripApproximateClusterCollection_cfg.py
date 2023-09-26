import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[2]))

process.testReadSiStripApproximateClusterCollection = cms.EDAnalyzer("TestReadSiStripApproximateClusterCollection",
    expectedIntegralValues = cms.vuint32(11, 21, 31, 41, 51, 62, 73),
    collectionTag = cms.InputTag("collectionProducer", "", "PROD")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSiStripApproximateClusterCollection2.root')
)

process.path = cms.Path(process.testReadSiStripApproximateClusterCollection)

process.endPath = cms.EndPath(process.out)
