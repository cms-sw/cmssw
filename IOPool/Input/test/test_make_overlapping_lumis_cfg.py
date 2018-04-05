import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.a = cms.EDProducer("IntProducer",
                             ivalue = cms.int32(10))

process.b = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(100*1000))

process.f = cms.EDFilter("ModuloEventIDFilter", modulo = cms.uint32(3), offset = cms.uint32(2) )

process.p = cms.Path(process.f+process.b, cms.Task(process.a) )

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("overlap.root"), 
                               outputCommands = cms.untracked.vstring("drop *", "keep *_b_*_*","keep *_a_*_*") )

process.prnt = cms.OutputModule("AsciiOutputModule", 
                               outputCommands = cms.untracked.vstring("drop *", "keep *_b_*_*","keep *_a_*_*") )

process.source = cms.Source("EmptySource", numberEventsInLuminosityBlock = cms.untracked.uint32(3))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.o = cms.EndPath(process.out)
process.o2 = cms.EndPath(process.prnt) 

process.options = cms.untracked.PSet( numberOfThreads = cms.untracked.uint32(3),
                                      numberOfStreams = cms.untracked.uint32(0),
                                      numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(2))
