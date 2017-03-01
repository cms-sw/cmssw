import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(0)
)

process.waiter = cms.EDProducer("WaitingThreadIntProducer",
                                streamsToAccumulate = cms.untracked.uint32(3),
#                                streamsToAccumulate = cms.untracked.uint32(4),
                                tags = cms.VInputTag("busy1","busy2")
                                )

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))
process.busy2 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(2), iterations = cms.uint32(10*1000*1000))

process.tester = cms.EDAnalyzer("IntTestAnalyzer",
                                moduleLabel = cms.untracked.string("waiter"),
                                valueMustMatch = cms.untracked.int32(5))

process.p = cms.Path(process.tester)
