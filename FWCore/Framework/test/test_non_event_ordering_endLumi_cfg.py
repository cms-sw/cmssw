import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.a = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(1))
process.b = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(2), consumesEndLuminosityBlock = cms.InputTag("c","endLumi") )
process.c = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(3), consumesEndLuminosityBlock = cms.InputTag("a", "endLumi"))

process.schedule = cms.Schedule(tasks=cms.Task(process.a,process.b,process.c))
