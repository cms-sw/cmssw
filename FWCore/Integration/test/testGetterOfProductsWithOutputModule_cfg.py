import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 3

process.thing = cms.EDProducer("ThingProducer",
    nThings = cms.int32(4)
)

process.testOne = cms.OutputModule("TestOutputWithGetterOfProducts",
    # 6 in one ThingCollection (0 + 1 + 2 + 3)
    # 6 * (3 writeEvent + 1 writeLumi * 2 (begin + end) + 1 writeRun * 2 (begin + end)) = 42 for writes
    # 6 * (1 endLumi * 2 (begin + end) + 1 endRun * 2 (begin + end)) = 24 for end transitions
    # 6 * (1 beginLumi * 1 (begin) + 1 beginRun * 2 (begin)) = 12 for begin transitions
    # 42 + 24 + 12 = 78
    expectedSum = cms.untracked.uint32(78)
)

process.testGlobal = cms.OutputModule("TestOutputWithGetterOfProductsGlobal",
    # 6 in one ThingCollection (0 + 1 + 2 + 3)
    # 6 * (3 writeEvent + 1 writeLumi * 2 (begin + end) + 1 writeRun * 2 (begin + end)) = 42 for writes
    # 6 * (1 endLumi * 2 (begin + end) + 1 endRun * 2 (begin + end)) = 24 for end transitions
    # 6 * (1 beginLumi * 1 (begin) + 1 beginRun * 2 (begin)) = 12 for begin transitions
    # 42 + 24 + 12 = 78
    expectedSum = cms.untracked.uint32(78)
)

process.testLimited = cms.OutputModule("TestOutputWithGetterOfProductsLimited",
    # 6 in one ThingCollection (0 + 1 + 2 + 3)
    # 6 * (3 writeEvent + 1 writeLumi * 2 (begin + end) + 1 writeRun * 2 (begin + end)) = 42 for writes
    # 6 * (1 endLumi * 2 (begin + end) + 1 endRun * 2 (begin + end)) = 24 for end transitions
    # 6 * (1 beginLumi * 1 (begin) + 1 beginRun * 2 (begin)) = 12 for begin transitions
    # 42 + 24 + 12 = 78
    expectedSum = cms.untracked.uint32(78)
)

process.path = cms.Path(process.thing)

process.endPath = cms.EndPath(process.testOne * process.testGlobal * process.testLimited)
