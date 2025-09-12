import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeSPLIT101.root',
        'file:testRunMergeSPLIT102.root',
        'file:testRunMergeSPLIT103.root',
        'file:testRunMergeSPLIT101.root',
        'file:testRunMergeSPLIT102.root',
        'file:testRunMergeSPLIT103.root'
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('41:11-41:20', '42:21-42:25'),
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
        10001,   10002,  10003,
        10001,   10002,  10003,
        10001,   10002,  10003,
        10001,   10002,  10003
    ),

    expectedEndRunProd = cms.untracked.vint32(
        100001,   100002,  100003,
        100001,   100002,  100003,
        100001,   100002,  100003,
        100001,   100002,  100003
    ),

    expectedEndRunProdImproperlyMerged = cms.untracked.vint32(
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        101,       102,    103,
        101,       102,    103,
        101,       102,    103,
        101,       102,    103,
        101,       102,    103,
        101,       102,    103,
        101,       102,    103,
        101,       102,    103,
        101,       102,    103,
        101,       102,    103
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003,
        1001,       1002,    1003
    ),

    expectedEndLumiProdImproperlyMerged = cms.untracked.vint32(
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0,
        0,   0,  0
    ),

    verbose = cms.untracked.bool(False)
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")
process.task = cms.Task(process.thingWithMergeProducer)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeTEST101.root')
)

process.e = cms.EndPath(process.test * process.out, process.task)
