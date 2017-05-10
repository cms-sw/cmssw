import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.vsimple = cms.EDProducer("VSimpleProducer",
    size = cms.int32(7)
)

process.tester = cms.EDProducer("AVSimpleProducer", src = cms.InputTag("vsimple"))

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("avtester.root"))

process.t = cms.Task(process.vsimple, process.tester)

process.o = cms.EndPath(process.out, process.t)
