import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.prod = cms.EDProducer("GenericProductProducer")

process.path = cms.Path(
    process.prod
)
