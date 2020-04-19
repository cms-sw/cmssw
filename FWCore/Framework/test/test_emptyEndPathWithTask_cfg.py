import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

# note that these modules get deleted, but the module dependence check is made first
process.intProducer1 = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.intProducer2 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1"))

process.ep = cms.EndPath(cms.Task(process.intProducer1, process.intProducer2))
