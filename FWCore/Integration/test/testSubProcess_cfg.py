import FWCore.ParameterSet.Config as cms
process = cms.Process("FIRST")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(30)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(4),
    numberEventsInRun = cms.untracked.uint32(10)
)

copyProcess = cms.Process("COPY")
process.subProcess = cms.SubProcess(copyProcess)

prodProcess = cms.Process("SECOND")
copyProcess.subProcess = cms.SubProcess(prodProcess)

prodProcess.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

prodProcess.p1 = cms.Path(prodProcess.thingWithMergeProducer)

copy2Process = cms.Process("COPY2")
prodProcess.subProcess = cms.SubProcess(copy2Process)

prod2Process = cms.Process("PROD")
copy2Process.subProcess = cms.SubProcess(prod2Process)

prod2Process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

# Reusing some code I used for testing merging, although in this
# context it has nothing to do with merging.
prod2Process.testmerge = cms.EDAnalyzer("TestMergeResults",

    expectedProcessHistoryInRuns = cms.untracked.vstring(
        'SECOND',            # Run 1
        'PROD',
        'SECOND',            # Run 2
        'PROD',
        'SECOND',            # Run 3
        'PROD'
    )
)

prod2Process.test = cms.EDAnalyzer('RunLumiEventAnalyzer',
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

prod2Process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSubProcess.root')
)

prod2Process.path1 = cms.Path(prod2Process.thingWithMergeProducer)

prod2Process.path2 = cms.Path(prod2Process.test*prod2Process.testmerge)

prod2Process.endPath1 = cms.EndPath(prod2Process.out)
