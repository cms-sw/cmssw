import FWCore.ParameterSet.Config as cms

process = cms.Process("SPLIT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge101.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('41:11-41:15', '42:11-42:15'),
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeSPLIT101.root')
)

process.e = cms.EndPath(process.out, process.task)
