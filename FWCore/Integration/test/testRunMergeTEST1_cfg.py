
# This is the same as testRunMergeTEST except the noEventSort
# option is set in the PoolSource, which changes the order
# events are processed.  Within a LuminosityBlock, they are
# in entry order instead of event number order.  The RunLumiEventAnalyzer
# module checks this.

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
        'file:testRunMerge.root',
        'file:testRunMerge.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
        'file:testRunMerge1.root', 
        'file:testRunMerge2.root', 
        'file:testRunMerge3.root',
        'file:testRunMerge4.root',
        'file:testRunMerge5.root'

    ),
    noEventSort = cms.untracked.bool(True)
    , duplicateCheckMode = cms.untracked.string('checkEachRealDataFile')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:testRunMergeRecombined1.root')
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
        10001,   30006,  10003,  # end run 1
        10001,   10002,  10003,  # end run 2
        10001,   20004,  10003,  # end run 11
        10001,   30006,  10003,  # end run 1
        10001,   10002,  10003,  # end run 2
        10001,   20004,  10003   # end run 11
    ),

    expectedEndRunProd = cms.untracked.vint32(
        100001, 300006, 100003,  # end run 1
        100001, 100002, 100003,  # end run 2
        100001, 200004, 100003,  # end run 11
        100001, 300006, 100003,  # end run 1
        100001, 100002, 100003,  # end run 2
        100001, 200004, 100003   # end run 11
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        101,       306,    103,  # end run 1 lumi 1
        101,       102,    103,  # end run 2 lumi 1
        101,       102,    103,  # end run 11 lumi 1
        101,       102,    103,  # end run 11 lumi 2
        101,       306,    103,  # end run 1 lumi 1
        101,       102,    103,  # end run 2 lumi 1
        101,       102,    103,  # end run 11 lumi 1
        101,       102,    103   # end run 11 lumi 2
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        1001,     3006,   1003,  # end run 1 lumi 1
        1001,     1002,   1003,  # end run 2 lumi 1
        1001,     1002,   1003,  # end run 11 lumi 1
        1001,     1002,   1003,  # end run 11 lumi 2
        1001,     3006,   1003,  # end run 1 lumi 1
        1001,     1002,   1003,  # end run 2 lumi 1
        1001,     1002,   1003,  # end run 11 lumi 1
        1001,     1002,   1003   # end run 11 lumi 2
    ),

    expectedBeginRunNew = cms.untracked.vint32(
        10001,   20004,  10003,  # end run 1
        10001,   10002,  10003,  # end run 2
        10001,   10002,  10003,  # end run 11
        10001,   20004,  10003,  # end run 1
        10001,   10002,  10003,  # end run 2
        10001,   10002,  10003   # end run 11
    ),

    expectedEndRunNew = cms.untracked.vint32(
        100001, 200004, 100003,  # end run 1
        100001, 100002, 100003,  # end run 2
        100001, 100002, 100003,  # end run 11
        100001, 200004, 100003,  # end run 1
        100001, 100002, 100003,  # end run 2
        100001, 100002, 100003   # end run 11
    ),

    expectedBeginLumiNew = cms.untracked.vint32(
        101,       204,    103,  # end run 1 lumi 1
        101,       102,    103,  # end run 2 lumi 1
        101,       102,    103,  # end run 11 lumi 1
        101,       102,    103,  # end run 11 lumi 2
        101,       204,    103,  # end run 1 lumi 1
        101,       102,    103,  # end run 2 lumi 1
        101,       102,    103,  # end run 11 lumi 1
        101,       102,    103   # end run 11 lumi 2
    ),

    expectedEndLumiNew = cms.untracked.vint32(
        1001,     2004,   1003,  # end run 1 lumi 1
        1001,     1002,   1003,  # end run 2 lumi 1
        1001,     1002,   1003,  # end run 11 lumi 1
        1001,     1002,   1003,  # end run 11 lumi 2
        1001,     2004,   1003,  # end run 1 lumi 1
        1001,     1002,   1003,  # end run 2 lumi 1
        1001,     1002,   1003,  # end run 11 lumi 1
        1001,     1002,   1003   # end run 11 lumi 2
    ),

    expectedDroppedEvent = cms.untracked.vint32(13, 10003, 100003, 103, 1003),
    verbose = cms.untracked.bool(False),

    expectedParents = cms.untracked.vstring(
        'm1', 'm1', 'm1', 'm1', 'm1',
        'm1', 'm1', 'm1', 'm1', 'm1',
        'm2', 'm2', 'm2', 'm2', 'm2',
        'm3', 'm3', 'm3', 'm3', 'm3',
        'm3', 'm3', 'm3', 'm3', 'm3',
        'm2', 'm2', 'm2', 'm2', 'm2',
        'm1', 'm1',
        'm1', 'm1', 'm1', 'm1', 'm1',
        'm1', 'm1', 'm1', 'm1', 'm1',
        'm2', 'm2', 'm2', 'm2', 'm2',
        'm3', 'm3', 'm3', 'm3', 'm3',
        'm3', 'm3', 'm3', 'm3', 'm3',
        'm2', 'm2', 'm2', 'm2', 'm2',
        'm1', 'm1'
   )
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
1, 1, 0,
1, 0, 0,
2, 0, 0,
2, 1, 0,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 0,
2, 0, 0,
11, 0, 0,
11, 1, 0,
11, 1, 1,
11, 1, 0,
11, 2, 0,
11, 2, 1,
11, 2, 0,
11, 0, 0,
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
1, 1, 0,
1, 0, 0,
2, 0, 0,
2, 1, 0,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 0,
2, 0, 0
)
)

process.path1 = cms.Path(process.test + process.test2)
process.endpath1 = cms.EndPath(process.out)
