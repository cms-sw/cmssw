import FWCore.ParameterSet.Config as cms

process = cms.Process("TRACING")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:file_for_grapher.root"))

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(100*1000*1000))
process.legacy1 = cms.EDProducer("BusyWaitIntLegacyProducer",ivalue = cms.int32(12), iterations = cms.uint32(3000*1000))

process.adder1 = cms.EDProducer("AddIntsProducer",
                                labels = cms.VInputTag("a","busy1","legacy1"))

process.busy2 = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(7), iterations = cms.uint32(4*1000*1000))
process.busy3 = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(2), iterations = cms.uint32(1000*1000))
process.legacy2 = cms.EDProducer("BusyWaitIntLegacyProducer", ivalue=cms.int32(-1), iterations = cms.uint32(6000*1000))

process.strt = cms.EDProducer("AddIntsProducer",
                              labels = cms.VInputTag("busy2","busy3","legacy2","b", "adder1"))

process.t = cms.Task(process.busy1, process.legacy1, process.adder1, process.busy2, process.busy3, process.legacy2)

process.p = cms.Path(process.strt, process.t)

process.options = cms.untracked.PSet( numberOfStreams = cms.untracked.uint32(4),
                                      numberOfThreads = cms.untracked.uint32(5))

process.add_(cms.Service("Tracer", printTimestamps = cms.untracked.bool(True)))
process.add_(cms.Service("StallMonitor", fileName = cms.untracked.string("stallMonitor.log")))
