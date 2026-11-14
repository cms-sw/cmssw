import FWCore.ParameterSet.Config as cms

process = cms.Process("SPLIT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#Contents of file
#testRunMerge101.root  "PROD" [run:41-42,lumi:11-20, ev:1-30]
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge101.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('41:16-41:20', '42:16-42:20'),
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeSPLIT102.root')
)
#Contents of file
#testRunMergeSPLIT102.root  "PROD" and "SPLIT" [run:41-42,lumi:16-20, ev:16-30] with "PROD" range[run:41-42,lumi:11-20]

process.e = cms.EndPath(process.out, process.task)
