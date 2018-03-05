import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(2),
    numberOfStreams = cms.untracked.uint32(0)
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2) )

process.filter = cms.EDFilter("ModuloStreamIDFilter",
                              modulo = cms.uint32(2),
                              offset = cms.uint32(1))

process.fail = cms.EDProducer("FailingProducer")

process.p = cms.Path(process.filter+process.fail)

#process.add_(cms.Service("Tracer"))
