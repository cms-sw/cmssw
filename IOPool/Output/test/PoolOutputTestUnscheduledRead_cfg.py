import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUTREAD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.analyzeOther = cms.EDAnalyzer("OtherThingAnalyzer")

# Test that the Run and Lumi products that we expect to be
# produced in the previous process are really there.
# It gets the products and checks that they contain
# the expected values.
# This tests that beginRun,beginLumi,endLumi, endRun
# get called for EDProducers with Unscheduled turned on.
process.test = cms.EDAnalyzer("TestMergeResults",
    expectedBeginRunNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        10001,   10002,  10003,   # * begin run 1
        10001,   10002,  10003,   # * events
        10001,   10002,  10003    # end run 1
    ),

    expectedEndRunNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        100001, 100002, 100003,   # begin run 1
        100001, 100002, 100003,   # * events
        100001, 100002, 100003    # * end run 1
    ),

    expectedBeginLumiNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        101,       102,    103,   # * begin run 1 lumi 1
        101,       102,    103,   # * events
        101,       102,    103    # end run 1 lumi 1
    ),

    expectedEndLumiNew = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        1001,     1002,   1003,   # begin run 1 lumi 1
        1001,     1002,   1003,   # * events
        1001,     1002,   1003    # * end run 1 lumi 1
    )
)

process.tst = cms.Path(process.analyzeOther+process.test)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:PoolOutputTestUnscheduled.root')
)



