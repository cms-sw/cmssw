import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(60)
)

# This will generate:
#   run 41 lumis 21-30 events 1-30
#   run 42 lumis 21-30 events 1-30
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(41),
    firstLuminosityBlock = cms.untracked.uint32(21),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    numberEventsInRun = cms.untracked.uint32(30)
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMerge102.root')
)

process.task = cms.Task(process.thingWithMergeProducer)

process.e = cms.EndPath(process.out, process.task)
