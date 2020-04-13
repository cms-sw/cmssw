import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.options.numberOfThreads = 2
process.options.numberOfStreams = 0

process.dummySync = cms.EDProducer("SonicDummyProducerSync",
    input = cms.int32(1),
    Client = cms.PSet(
        factor = cms.int32(-1),
        wait = cms.int32(10),
        allowedTries = cms.uint32(0),
        fails = cms.uint32(0),
    ),
)

process.dummyPseudoAsync = cms.EDProducer("SonicDummyProducerPseudoAsync",
    input = cms.int32(2),
    Client = cms.PSet(
        factor = cms.int32(2),
        wait = cms.int32(10),
        allowedTries = cms.uint32(0),
        fails = cms.uint32(0),
    ),
)

process.dummyAsync = cms.EDProducer("SonicDummyProducerAsync",
    input = cms.int32(3),
    Client = cms.PSet(
        factor = cms.int32(5),
        wait = cms.int32(10),
        allowedTries = cms.uint32(0),
        fails = cms.uint32(0),
    ),
)

process.dummySyncRetry = process.dummySync.clone(
    Client = dict(
        wait = 2,
        allowedTries = 2,
        fails = 1,
    )
)

process.dummyPseudoAsyncRetry = process.dummyPseudoAsync.clone(
    Client = dict(
        wait = 2,
        allowedTries = 2,
        fails = 1,
    )
)

process.dummyAsyncRetry = process.dummyAsync.clone(
    Client = dict(
        wait = 2,
        allowedTries = 2,
        fails = 1,
    )
)

process.task = cms.Task(
    process.dummySync,process.dummyPseudoAsync,process.dummyAsync,
    process.dummySyncRetry,process.dummyPseudoAsyncRetry,process.dummyAsyncRetry,
)

process.testerSync = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(-1),
    moduleLabel = cms.untracked.string("dummySync"),
)

process.testerPseudoAsync = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(4),
    moduleLabel = cms.untracked.string("dummyPseudoAsync"),
)

process.testerAsync = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(15),
    moduleLabel = cms.untracked.string("dummyAsync"),
)

process.testerSyncRetry = process.testerSync.clone(
    moduleLabel = cms.untracked.string("dummySyncRetry")
)

process.testerPseudoAsyncRetry = process.testerPseudoAsync.clone(
    moduleLabel = cms.untracked.string("dummyPseudoAsyncRetry")
)

process.testerAsyncRetry = process.testerAsync.clone(
    moduleLabel = cms.untracked.string("dummyAsyncRetry")
)

process.p1 = cms.Path(process.testerSync, process.task)
process.p2 = cms.Path(process.testerPseudoAsync, process.task)
process.p3 = cms.Path(process.testerAsync, process.task)
process.p4 = cms.Path(process.testerSyncRetry, process.task)
process.p5 = cms.Path(process.testerPseudoAsyncRetry, process.task)
process.p6 = cms.Path(process.testerAsyncRetry, process.task)
