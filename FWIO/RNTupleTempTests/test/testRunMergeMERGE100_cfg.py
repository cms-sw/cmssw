import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#Contents of files
#testRunMergeSPLIT100.root  "PROD" and "SPLIT" [run:41-42,lumi:6-10, ev:16-30] with "PROD" range[run:41-42,lumi:1-10]
#testRunMergeSPLIT101.root  "PROD" and "SPLIT" [run:41-42,lumi:11-15, ev:1-15] with "PROD" range[run:41-42,lumi:11-20]

process.source = cms.Source("RNTupleTempSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeSPLIT100.root',
        'file:testRunMergeSPLIT101.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('41:6-41:15'),
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.test = cms.EDAnalyzer("TestMergeResults",

    #   Check to see that the value we read matches what we know
    #   was written. Expected values listed below come in sets of three
    #      value expected in Thing
    #      value expected in ThingWithMerge
    #      value expected in ThingWithIsEqual
    #   Each set of 3 is tested at endRun for the expected
    #   run values or at endLuminosityBlock for the expected
    #   lumi values. And then the next set of three values
    #   is tested at the next endRun or endLuminosityBlock.
    #   When the sequence of parameter values is exhausted it stops checking

    expectedBeginRunProd = cms.untracked.vint32(
        10001,   20004,  10003
    ),

    expectedEndRunProd = cms.untracked.vint32(
        100001,   200004,  100003
    ),

    expectedEndRunProdImproperlyMerged = cms.untracked.vint32(
        0,   0,  0
    )
)

process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('testRunMergeMERGE100.root')
)
#Contents of file
#testRunMergeMERGE100.root  "PROD", "SPLIT" & "MERGE" [run:41,lumi:6-15, ev:1-30]

process.e = cms.EndPath(process.test * process.out, process.task)
