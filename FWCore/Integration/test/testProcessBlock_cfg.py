import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.thingProducer = cms.EDProducer("ThingProducer")

process.path = cms.Path(process.thingProducer)

