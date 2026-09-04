import FWCore.ParameterSet.Config as cms

process = cms.Process("SPLIT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#Contents of file
#testRunMerge100.root  "PROD" [run:41-42,lumi:1-10, ev:1-30]
process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge100.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('41:6-41:10', '42:6-42:10'),
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testRunMergeSPLIT100.root')
)
#testRunMergeSPLIT100.root  "PROD" and "SPLIT" [run:41-42,lumi:6-10, ev:16-30] with "PROD" range[run:41-42,lumi:1-10]

process.e = cms.EndPath(process.out, process.task)
