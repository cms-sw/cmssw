import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))

process.testReadVectorDetId = cms.EDAnalyzer("TestReadVectorDetId",
    expectedTestValue = cms.uint32(21),
    collectionTag = cms.InputTag("collectionProducer", "", "PROD")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testVectorDetId2.root')
)

process.path = cms.Path(process.testReadVectorDetId)

process.endPath = cms.EndPath(process.out)
