import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIFollower")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource()

process.maxEvents.input = -1

process.receiver = MPIReceiver(
    upstream = "source",
    instance = 42,
    products = [
        dict(
            type = "FEDRawDataCollection",
            label = "fedRawDataCollectionProducer"
        ),
        dict(
            type = "RawDataBuffer",
            label = "rawDataBufferProducer"
        ),
        dict(
            type = "edm::TriggerResults",
            label = "triggerResultsProducer"
        ),
        dict(
            type = "trigger::TriggerEvent",
            label = "triggerEventProducer"
        )
    ]
)

# Phase-1 FED RAW data collection pseudo object
process.testReadFEDRawDataCollection = cms.EDAnalyzer("TestReadFEDRawDataCollection",
    fedRawDataCollectionTag = cms.InputTag("receiver", "fedRawDataCollectionProducer"),
    expectedFEDData0 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7),
    expectedFEDData3 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107)
)

# Phase-2 RAW data buffer pseudo object
process.testReadRawDataBuffer = cms.EDAnalyzer("TestReadRawDataBuffer",
    rawDataBufferTag = cms.InputTag("receiver", "rawDataBufferProducer"),
    dataPattern1 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    dataPattern2 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115)
)

# HLT trigger event pseudo object
process.testReadTriggerEvent = cms.EDAnalyzer("TestReadTriggerEvent",
    expectedUsedProcessName = cms.string("testName"),
    expectedCollectionTags = cms.vstring('moduleA', 'moduleB', 'moduleC'),
    expectedCollectionKeys = cms.vuint32(11, 21, 31),
    expectedIds = cms.vint32(1, 3, 5),
    # stick to values exactly convertible from double to float
    # to avoid potential rounding issues in the test, because
    # the configuration only supports double not float and
    # the data format holds floats.
    expectedPts = cms.vdouble(11.0, 21.0, 31.0),
    expectedEtas = cms.vdouble(101.0, 102.0, 103.0),
    expectedPhis = cms.vdouble(201.0, 202.0, 203.0),
    expectedMasses = cms.vdouble(301.0, 302.0, 303.0),
    expectedFilterTags = cms.vstring('moduleAA', 'moduleBB'),
    expectedElementsPerVector = cms.uint32(2),
    expectedFilterIds = cms.vint32(1001, 1002, 1003, 1004),
    expectedFilterKeys = cms.vuint32(2001, 2002, 2003, 2004),
    triggerEventTag = cms.InputTag("receiver", "triggerEventProducer")
)

# EDM trigger results pseudo object
process.testReadTriggerResults = cms.EDAnalyzer("TestReadTriggerResults",
    triggerResultsTag = cms.InputTag("receiver", "triggerResultsProducer"),
        expectedParameterSetID = cms.string('8b99d66b6c3865c75e460791f721202d'),
        expectedNames = cms.vstring(),
        expectedHLTStates = cms.vuint32(0, 1, 2, 3),
        expectedModuleIndexes = cms.vuint32(11, 21, 31, 41)
)

process.path = cms.Path(
    process.receiver +
    process.testReadFEDRawDataCollection +
    process.testReadRawDataBuffer +
    process.testReadTriggerEvent +
    process.testReadTriggerResults
)
