import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options = cms.untracked.PSet(
                            numberOfStreams = cms.untracked.uint32(3),
                            numberOfThreads = cms.untracked.uint32(3)
)

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.d = cms.EDProducer("NonEventIntProducer",
                           ivalue = cms.int32(1))

process.b = cms.EDProducer("NonEventIntProducer",
                           ivalue = cms.int32(2),
                           sleepTime = cms.uint32(3000),
                           consumesBeginProcessBlock = cms.InputTag("c","beginProcessBlock"),
                           expectBeginProcessBlock = cms.untracked.int32(3),
                           consumesBeginRun = cms.InputTag("c","beginRun"),
                           expectBeginRun = cms.untracked.int32(3),
                           consumesBeginLuminosityBlock = cms.InputTag("c","beginLumi"),
                           expectBeginLuminosityBlock = cms.untracked.int32(3),
                           consumesEndLuminosityBlock = cms.InputTag("c","endLumi"),
                           expectEndLuminosityBlock = cms.untracked.int32(3),
                           consumesEndRun = cms.InputTag("c","endRun"),
                           expectEndRun = cms.untracked.int32(3),
                           consumesEndProcessBlock = cms.InputTag("c","endProcessBlock"),
                           expectEndProcessBlock = cms.untracked.int32(3)
)

process.c = cms.EDProducer("NonEventIntProducer",
                           ivalue = cms.int32(3),
                           sleepTime = cms.uint32(3000),
                           consumesBeginProcessBlock = cms.InputTag("d","beginProcessBlock"),
                           expectBeginProcessBlock = cms.untracked.int32(1),
                           consumesBeginRun = cms.InputTag("d","beginRun"),
                           expectBeginRun = cms.untracked.int32(1),
                           consumesBeginLuminosityBlock = cms.InputTag("d","beginLumi"),
                           expectBeginLuminosityBlock = cms.untracked.int32(1),
                           consumesEndLuminosityBlock = cms.InputTag("d","endLumi"),
                           expectEndLuminosityBlock = cms.untracked.int32(1),
                           consumesEndRun = cms.InputTag("d","endRun"),
                           expectEndRun = cms.untracked.int32(1),
                           consumesEndProcessBlock = cms.InputTag("d", "endProcessBlock"),
                           expectEndProcessBlock = cms.untracked.int32(1)
)

process.t = cms.Task(process.d, process.c)
process.p = cms.Path(process.b, process.t)
