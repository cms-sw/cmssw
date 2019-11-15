import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.a = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(1))
process.b = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(2), consumesBeginLuminosityBlock = cms.InputTag("c","beginLumi") )
process.c = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(3), consumesBeginLuminosityBlock = cms.InputTag("a", "beginLumi"))

process.schedule = cms.Schedule(tasks=cms.Task(process.a,process.b,process.c))
