import sys
import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))

process.testReadHostVertexGNNSoA = cms.EDAnalyzer("TestReadHostVertexGNNSoA",
    input = cms.InputTag("vertexGNNSoA", "", "WRITE")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testVertexGNNSoAReader.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.testReadHostVertexGNNSoA)

process.endPath = cms.EndPath(process.out)
