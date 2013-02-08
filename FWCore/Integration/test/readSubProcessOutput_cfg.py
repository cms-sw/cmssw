import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:testSubProcess.root")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
      'readSubprocessOutput.root'
    )
)


# Reusing some code I used for testing merging, although in this
# context it has nothing to do with merging.
# Here we are checking the event, run, and lumi products
# from the last subprocess in the chain of subprocesses
# are there.
process.testproducts = cms.EDAnalyzer("TestMergeResults",

    expectedBeginRunProd = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        10001,   10002,  10003,   # * begin run 1
        10001,   10002,  10003,   # * events
        10001,   10002,  10003,   # end run 1
        10001,   10002,  10003,   # * begin run 2
        10001,   10002,  10003,   # * events
        10001,   10002,  10003,   # end run 2
        10001,   10002,  10003,   # * begin run 2
        10001,   10002,  10003,   # * events
        10001,   10002,  10003    # end run 3
    ),

    expectedEndRunProd = cms.untracked.vint32(
        0,           0,      0,      # start
        0,           0,      0,      # begin file 1
        100001,   100002,  100003,   # begin run 1
        100001,   100002,  100003,   # * events
        100001,   100002,  100003,   # * end run 1
        100001,   100002,  100003,   # begin run 2
        100001,   100002,  100003,   # * events
        100001,   100002,  100003,   # * end run 2
        100001,   100002,  100003,   # begin run 2
        100001,   100002,  100003,   # * events
        100001,   100002,  100003    # * end run 3
    ),

    expectedBeginLumiProd = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        101,       102,    103,   # * begin run 1 lumi 1
        101,       102,    103,   # * events
        101,       102,    103    # end run 1 lumi 1
# There are more, but all with the same pattern as the first        
    ),

    expectedEndLumiProd = cms.untracked.vint32(
        0,           0,      0,   # start
        0,           0,      0,   # begin file 1
        1001,     1002,   1003,   # begin run 1 lumi 1
        1001,     1002,   1003,   # * events
        1001,     1002,   1003    # * end run 1 lumi 1
    ),

    expectedProcessHistoryInRuns = cms.untracked.vstring(
        'SECOND',            # Run 1
        'PROD',
        'READ',
        'SECOND',            # Run 2
        'PROD',
        'READ',
        'SECOND',            # Run 3
        'PROD',
        'READ'
    ),
    verbose = cms.untracked.bool(True)
)

process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
1,   0,   0,
1,   1,   0,
1,   1,   1,
1,   1,   2,
1,   1,   3,
1,   1,   4,
1,   1,   0,
1,   2,   0,
1,   2,   5,
1,   2,   6,
1,   2,   7,
1,   2,   8,
1,   2,   0,
1,   3,   0,
1,   3,   9,
1,   3,   10,
1,   3,   0,
1,   0,   0,
2,   0,   0,
2,   1,   0,
2,   1,   1,
2,   1,   2,
2,   1,   3,
2,   1,   4,
2,   1,   0,
2,   2,   0,
2,   2,   5,
2,   2,   6,
2,   2,   7,
2,   2,   8,
2,   2,   0,
2,   3,   0,
2,   3,   9,
2,   3,   10,
2,   3,   0,
2,   0,   0,
3,   0,   0,
3,   1,   0,
3,   1,   1,
3,   1,   2,
3,   1,   3,
3,   1,   4,
3,   1,   0,
3,   2,   0,
3,   2,   5,
3,   2,   6,
3,   2,   7,
3,   2,   8,
3,   2,   0,
3,   3,   0,
3,   3,   9,
3,   3,   10,
3,   3,   0,
3,   0,   0
)
)

process.path1 = cms.Path(process.test*process.testproducts)

process.ep = cms.EndPath(process.out)
