# A test of the noRunLumiSort configuration parameter
# where the input has non-contiguous event sequences
# from the same run. This configuration reads a file
# created using that parameter and checks that the
# run product and lumi product merging that occurred
# was done properly.

# It is expected there are 8 warnings that
# print out while this runs related to merging.
# The test should pass with these warnings.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMergeNoRunLumiSort.root'
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
    #   0's are just placeholders, if the value is a "0" the check is not made.

    expectedBeginRunProd = cms.untracked.vint32(
        0,   60012,  10003,
        0,   20004,  10003
    ),
                            
    expectedEndRunProd = cms.untracked.vint32(
        0, 600012, 100003,
        0, 200004, 100003
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        0,       612,    103,
        0,       204,    103
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        0,     6012,   1003,
        0,     2004,   1003
    ),

    expectedBeginRunNew = cms.untracked.vint32(
        10001,   40008,  10003,
        10001,   20004,  10003
    ),

    expectedEndRunNew = cms.untracked.vint32(
        100001, 400008, 100003,
        100001, 200004, 100003
    ),

    expectedBeginLumiNew = cms.untracked.vint32(
        101,       408,    103,
        101,       204,    103
    ),

    expectedEndLumiNew = cms.untracked.vint32(
        1001,     4008,   1003,
        1001,     2004,   1003
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
1, 0, 0
)
)
process.test2.expectedRunLumiEvents.extend([
2, 0, 0,
2, 1, 0,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 1,
2, 1, 2,
2, 1, 3,
2, 1, 4,
2, 1, 5,
2, 1, 0,
2, 0, 0
])

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeTEST6.root')
)

process.path1 = cms.Path(process.test * process.test2)
process.e = cms.EndPath(process.out)
