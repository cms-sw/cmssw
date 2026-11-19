import FWCore.ParameterSet.Config as cms

process = cms.Process("SPLIT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#Contents of file
#testRunMerge101.root  "PROD" [run:41-42,lumi:11-20, ev:1-30]
process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge101.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('41:11-41:15', '42:11-42:15'),
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testRunMergeSPLIT101.root')
)
#Contents of file
#testRunMergeSPLIT101.root  "PROD" and "SPLIT" [run:41-42,lumi:11-15, ev:1-15] with "PROD" range[run:41-42,lumi:11-20]


process.e = cms.EndPath(process.out, process.task)
