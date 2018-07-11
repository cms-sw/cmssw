import FWCore.ParameterSet.Config as cms

process = cms.Process("COPY")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeTEST100.root'
    ),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

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
        10001,   40008,  10003,
        10001,   20004,  10003
    ),

    expectedEndRunProd = cms.untracked.vint32(
        100001,   400008,  100003,
        100001,   200004,  100003
    ),

    expectedEndRunProdImproperlyMerged = cms.untracked.vint32(
        0,   1,  0,
        0,   0,  0
    )
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeCOPY100.root')
)

process.e = cms.EndPath(process.test * process.out, process.task)
