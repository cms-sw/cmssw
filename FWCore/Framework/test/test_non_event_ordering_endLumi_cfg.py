import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.d = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(1))
process.b = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(2),
                           consumesEndLuminosityBlock = cms.InputTag("c","endLumi"),
                           expectEndLuminosityBlock = cms.untracked.int32(3) )
process.c = cms.EDProducer("NonEventIntProducer", ivalue = cms.int32(3),
                           consumesEndLuminosityBlock = cms.InputTag("d", "endLumi"),
                           expectEndLuminosityBlock = cms.untracked.int32(1))

process.t = cms.Task(process.d, process.c)
process.p = cms.Path(process.b, process.t)
