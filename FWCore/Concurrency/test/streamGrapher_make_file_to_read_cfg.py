import FWCore.ParameterSet.Config as cms

process =cms.Process("FILE")

process.source = cms.Source("EmptySource")

process.a = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
process.b = cms.EDProducer("IntProducer", ivalue = cms.int32(3))

process.o = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("file_for_grapher.root"))

process.ep = cms.EndPath(process.o)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))
process.options = cms.untracked.PSet(allowUnscheduled = cms.untracked.bool(True) )
