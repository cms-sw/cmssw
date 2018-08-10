import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(32)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root',
        'file:testRunMerge4.root',
        'file:testRunMerge5.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_C_*_*',
        'drop edmtestThingWithMerge_makeThingToBeDropped1_*_*'
    )
    , duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

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
    #   0's are just placeholders, if the value is a "0" the check is not made.

    expectedBeginRunProd = cms.untracked.vint32(
        10001,   20004,  10003,   # File boundary before this causing merge
        10001,   10002,  10003,
        10001,   10002,  10004
    ),
                            
    expectedEndRunProd = cms.untracked.vint32(
        100001, 200004, 100003,   # File boundary before this causing merge
        100001, 100002, 100003,
        100001, 100002, 100004
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        101,       204,    103,   # File boundary before this causing merge
        101,       102,    103,
        101,       102,    104
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        1001,     2004,   1003,   # File boundary before this causing merge
        1001,     1002,   1003,
        1001,     1002,   1004
    ),

    expectedBeginRunNew = cms.untracked.vint32(
        10001,   10002,  10003,
        10001,   10002,  10003,
        10001,   10002,  10003
    ),

    expectedEndRunNew = cms.untracked.vint32(
        100001, 100002, 100003,
        100001, 100002, 100003,
        100001, 100002, 100003
    ),

    expectedBeginLumiNew = cms.untracked.vint32(
        101,       102,    103,
        101,       102,    103,
        101,       102,    103
    ),

    expectedEndLumiNew = cms.untracked.vint32(
        1001,     1002,   1003,
        1001,     1002,   1003,
        1001,     1002,   1003
    ),

    expectedDroppedEvent1 = cms.untracked.vint32(13, -1, -1, -1, 13),

    expectedRespondToOpenInputFile = cms.untracked.int32(5),
    expectedRespondToCloseInputFile = cms.untracked.int32(5),

    expectedInputFileNames = cms.untracked.vstring(
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root',
        'file:testRunMerge4.root',
        'file:testRunMerge5.root'
    ),

    verbose = cms.untracked.bool(False),
    testAlias = cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMerge.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_makeThingToBeDropped_*_*',
        'drop *_aliasForThingToBeDropped2_*_*',
        'drop *_B_*_*',
        'drop *_G_*_*',
        'drop *_H_*_*',
        'drop *_I_*_*',
        'drop *_J_*_*'
    )
)

process.checker = cms.OutputModule("GetProductCheckerOutputModule")

process.path1 = cms.Path(process.thingWithMergeProducer + process.test)
process.e = cms.EndPath(process.out * process.checker)
