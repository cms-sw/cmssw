import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.intProducer1 = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.intProducer2 = cms.EDProducer("AddIntsProducer", labels = cms.vstring("intProducer1"))

process.p = cms.Path(cms.Task(process.intProducer1, process.intProducer2))
