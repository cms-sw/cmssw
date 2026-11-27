import sys
import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 5

process.vertexGNNSoA = cms.EDProducer("TestWriteHostVertexGNNSoA",
    trackSize = cms.uint32(500)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(sys.argv[1])
)

process.path = cms.Path(process.vertexGNNSoA)
process.endPath = cms.EndPath(process.out)
