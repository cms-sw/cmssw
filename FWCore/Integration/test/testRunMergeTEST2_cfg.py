#
# Test the most standard case where we read the same run
#  from different files but each file has its own lumi blocks
#

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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge4.root',
        'file:testRunMerge6.root'
    )
)

process.test = cms.EDAnalyzer("TestMergeResults",

    #   These values below are just arbitrary and meaningless
    #   We are checking to see that the value we get out matches what
    #   was put in.

    #   expected values listed below come in sets of three
    #      value expected in Thing
    #      value expected in ThingWithMerge
    #      value expected in ThingWithIsEqual
    #   This set of 3 is repeated below at each point it might change
    #   The Prod suffix refers to objects from the process named PROD
    #   The New suffix refers to objects created in the most recent process
    #   When the sequence of parameter values is exhausted it stops checking
    #   0's are just placeholders, if the value is a "0" the check is not made
    #   and it indicates the product does not exist at that point.
    #   *'s indicate lines where the checks are actually run by the test module.

    expectedBeginRunProd = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        10001,   10002,  10003,  # * begin run 11
        10001,   20004,  10003,  # end run 11
        10001,   10002,  10003,  # * begin run 11
        10001,   10002,  10003   # end run 11
    ),

    expectedEndRunProd = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        100001, 100002, 100003,  # * begin run 11
        100001, 200004, 100003,  # * end run 11
        100001, 200004, 100003,  # * begin run 11
        100001, 100002, 100003   # * end run 11
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        101,       102,    103,  # * begin run 11 lumi 1
        101,       102,    103,  # end run 11 lumi 1
        101,       102,    103,  # * begin run 11 lumi 2
        101,       102,    103,  # end run 11 lumi 2
        101,       102,    103,  # * begin run 11 lumi 3
        101,       102,    103,  # end run 11 lumi 3
        101,       102,    103   # * begin run 11 lumi 4
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        1001,     1002,   1003,  # * begin run 11 lumi 1
        1001,     3006,   1003,  # * end run 11 lumi 1
        1001,     1002,   1003,  # * begin run 11 lumi 2
        1001,     1002,   1003,  # * end run 11 lumi 2
        1001,     1002,   1003,  # * begin run 11 lumi 3
        1001,     1002,   1003,  # * end run 11 lumi 3
        1001,     1002,   1003   # * begin run 11 lumi 4
    ),

    expectedBeginRunNew = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        10001,   10002,  10003,  # * begin run 11
        10001,   20004,  10003   # end run 11
    ),

    expectedEndRunNew = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        100001, 100002, 100003,  # * begin run 11
        100001, 200004, 100003,  # * end run 11
        100001, 200004, 100003   # * begin run 2
    ),

    expectedBeginLumiNew = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        101,       102,    103,  # * begin run 11 lumi 1
        101,       204,    103,  # end run 11 lumi 1
        101,       102,    103,  # * begin run 11 lumi 2
        101,       102,    103,  # end run 11 lumi 2
        101,       102,    103,  # * begin run 11 lumi 3
        101,       102,    103,  # end run 11 lumi 3
        101,       102,    103   # * begin run 11 lumi 4
    ),

    expectedEndLumiNew = cms.untracked.vint32(
        0,           0,      0,  # start
        0,           0,      0,  # begin file 1
        1001,     1002,   1003,  # * begin run 11 lumi 1
        1001,     2004,   1003,  # * end run 11 lumi 1
        1001,     1002,   1003,  # * begin run 11 lumi 2
        1001,     1002,   1003,  # * end run 11 lumi 2
        1001,     1002,   1003,  # * begin run 11 lumi 3
        1001,     1002,   1003,  # * end run 11 lumi 3
        1001,     1002,   1003   # * begin run 11 lumi 4
    ),

    verbose = cms.untracked.bool(False) #,

)

process.test2 = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
11, 0, 0,
11, 1, 0,
11, 1, 1,
11, 1, 0,
11, 2, 0,
11, 2, 2,
11, 2, 3,
11, 2, 4,
11, 2, 0,
11, 3, 0,
11, 3, 5,
11, 3, 6,
11, 3, 7,
11, 3, 0,
11, 4, 0,
11, 4, 8,
11, 4, 9,
11, 4, 10,
11, 4, 0,
11, 0, 0
)
)

process.path1 = cms.Path(process.test + process.test2)

