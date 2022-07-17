# The purpose of this configuration is to prepare
# input for a test of the noRunLumiSort configuration
# parameter that has non-contiguous sequences of
# events from the same run (and the same lumi).

# It is expected there are 6 warnings that
# print out while this runs related to merging.
# The test should pass with these warnings.

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testRunMerge1.root',
        'file:testRunMerge2.root',
        'file:testRunMerge3.root'
    ),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_A_*_*',
        'drop *_B_*_*',
        'drop *_C_*_*',
        'drop *_D_*_*',
        'drop *_E_*_*',
        'drop *_F_*_*',
        'drop *_G_*_*',
        'drop *_H_*_*',
        'drop *_I_*_*',
        'drop *_J_*_*',
        'drop *_K_*_*',
        'drop *_L_*_*',
        'drop *_tryNoPut_*_*',
        'drop *_aliasForThingToBeDropped2_*_*',
        'drop *_dependsOnThingToBeDropped1_*_*',
        'drop *_makeThingToBeDropped_*_*',
        'drop edmtestThingWithMerge_makeThingToBeDropped1_*_*',
        'drop edmtestThing_*_*_*'
    )
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
        0,   20004,  10003,   # File boundary before this causing merge
        0,   10002,  10003,
        0,   10002,  10004
    ),
    expectedEndRunProd = cms.untracked.vint32(
        0, 200004, 100003,   # File boundary before this causing merge
        0, 100002, 100003,
        0, 100002, 100004
    ),
    expectedBeginLumiProd = cms.untracked.vint32(
        0,       204,    103,   # File boundary before this causing merge
        0,       102,    103,
        0,       102,    104
    ),
    expectedEndLumiProd = cms.untracked.vint32(
        0,     2004,   1003,   # File boundary before this causing merge
        0,     1002,   1003,
        0,     1002,   1004
    ),
    expectedBeginRunNew = cms.untracked.vint32(
        0,   10002,  10003,
        0,   10002,  10003,
        0,   10002,  10003
    ),
    expectedEndRunNew = cms.untracked.vint32(
        0,   100002,  100003,
        0,   100002,  100003,
        0,   100002,  100003
    ),
    expectedBeginLumiNew = cms.untracked.vint32(
        0,   102,  103,
        0,   102,  103,
        0,   102,  103
    ),
    expectedEndLumiNew = cms.untracked.vint32(
        0,   1002,  1003,
        0,   1002,  1003,
        0,   1002,  1003
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRunMergeMERGE6.root')
)

process.path1 = cms.Path(process.thingWithMergeProducer + process.test)
process.e = cms.EndPath(process.out)
