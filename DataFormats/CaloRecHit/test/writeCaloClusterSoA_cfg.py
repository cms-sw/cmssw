# Writes a reco::CaloClusterHostCollection to a TTree file and to an RNTuple file.
import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 8

process.writer = cms.EDProducer("TestWriteCaloClusterSoA")

process.outputTTree = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("testCaloClusterSoA.root")
)
process.outputRNTuple = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string("testCaloClusterSoA_rntuple.root")
)

process.path = cms.Path(process.writer)
process.endPath = cms.EndPath(process.outputTTree + process.outputRNTuple)
