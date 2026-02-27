import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))

process.testReadHostVertexSoA = cms.EDAnalyzer("TestReadHostVertexSoA",
    input = cms.InputTag("vertexSoA", "", "WRITE")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testVertexSoAReader.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.testReadHostVertexSoA)

process.endPath = cms.EndPath(process.out)

