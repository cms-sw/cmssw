import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.threshold = 'ERROR'

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  fileMode  = cms.untracked.string('FULLMERGE'),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeMERGE4.root',
        'file:testRunMergeMERGE4.root'
    )
    , duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
    , noEventSort = cms.untracked.bool(True)
    , lumisToProcess = cms.untracked.VLuminosityBlockRange('1:1')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:testRunMergeRecombined4.root')
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
    #   0's are just placeholders, if the value is a "0" the check is not made.

    expectedBeginRunProd = cms.untracked.vint32(
        10001,   80016,  10003   # end run 1
    ),

    expectedEndRunProd = cms.untracked.vint32(
        100001,  800016, 100003   # end run 1
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        101,  816, 103  # end run 1 lumi 1
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        1001,  8016, 1003   # end run 1 lumi 1
    ),

    expectedBeginRunNew = cms.untracked.vint32(
        10001,   60012,  10003   # end run 1
    ),

    expectedEndRunNew = cms.untracked.vint32(
        100001,  600012, 100003   # end run 1
    ),

    expectedBeginLumiNew = cms.untracked.vint32(
        101,  612, 103   # end run 1 lumi 1
    ),

    expectedEndLumiNew = cms.untracked.vint32(
        1001,  6012, 1003   # end run 1 lumi 1
    ),

    verbose = cms.untracked.bool(True)
)

process.test2 = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
1, 0, 0,
1, 1, 0,
1, 1, 11,
1, 1, 12,
1, 1, 13,
1, 1, 14,
1, 1, 15,
1, 1, 16,
1, 1, 17,
1, 1, 18,
1, 1, 19,
1, 1, 20,
1, 1, 21,
1, 1, 22,
1, 1, 23,
1, 1, 24,
1, 1, 25,
1, 1, 1,
1, 1, 2,
1, 1, 3,
1, 1, 4,
1, 1, 5,
1, 1, 6,
1, 1, 7,
1, 1, 8,
1, 1, 9,
1, 1, 10,
1, 1, 26,
1, 1, 11,
1, 1, 12,
1, 1, 13,
1, 1, 14,
1, 1, 15,
1, 1, 16,
1, 1, 17,
1, 1, 18,
1, 1, 19,
1, 1, 20,
1, 1, 21,
1, 1, 22,
1, 1, 23,
1, 1, 24,
1, 1, 25,
1, 1, 1,
1, 1, 2,
1, 1, 3,
1, 1, 4,
1, 1, 5,
1, 1, 6,
1, 1, 7,
1, 1, 8,
1, 1, 9,
1, 1, 10,
1, 1, 26,
1, 1, 0,
1, 0, 0
)
)

process.path1 = cms.Path(process.test + process.test2)
process.endpath1 = cms.EndPath(process.out)
