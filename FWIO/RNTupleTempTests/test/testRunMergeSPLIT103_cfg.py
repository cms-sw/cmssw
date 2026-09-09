import FWCore.ParameterSet.Config as cms

process = cms.Process("SPLIT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#Contents of file
#testRunMerge102.root  "PROD" [run:41-42,lumi:21-30, ev:1-30]
process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge102.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('41:21-41:25', '42:21-42:25'),
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testRunMergeSPLIT103.root')
)
#Contents of file
#testRunMergeSPLIT103.root  "PROD" and "SPLIT" [run:41-42,lumi:21-25, ev:1-15] with "PROD" range[run:41-42,lumi:21-30]

process.e = cms.EndPath(process.out, process.task)
