import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIServer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
# MPIController supports a single concurrent LuminosityBlock
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from HeterogeneousCore.MPICore.modules import *

process.mpiController = MPIController(
    mode = 'CommWorld'
)

# Phase-1 FED RAW data collection pseudo object
process.fedRawDataCollectionProducer = cms.EDProducer("TestWriteFEDRawDataCollection",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    FEDData0 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7),
    FEDData3 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107)
)

# Phase-2 RAW data buffer pseudo object
process.rawDataBufferProducer = cms.EDProducer("TestWriteRawDataBuffer",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    dataPattern1 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    dataPattern2 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115)
)

# HLT trigger event pseudo object
process.triggerEventProducer = cms.EDProducer("TestWriteTriggerEvent",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    usedProcessName = cms.string("testName"),
    collectionTags = cms.vstring('moduleA', 'moduleB', 'moduleC'),
    collectionKeys = cms.vuint32(11, 21, 31),
    ids = cms.vint32(1, 3, 5),
    # stick to values exactly convertible from double to float
    # to avoid potential rounding issues in the test, because
    # the configuration only supports double not float and
    # the data format holds floats.
    pts = cms.vdouble(11.0, 21.0, 31.0),
    etas = cms.vdouble(101.0, 102.0, 103.0),
    phis = cms.vdouble(201.0, 202.0, 203.0),
    masses = cms.vdouble(301.0, 302.0, 303.0),
    filterTags = cms.vstring('moduleAA', 'moduleBB'),
    elementsPerVector = cms.uint32(2),
    filterIds = cms.vint32(1001, 1002, 1003, 1004),
    filterKeys = cms.vuint32(2001, 2002, 2003, 2004)
)

# EDM trigger results pseudo object
process.triggerResultsProducer = cms.EDProducer("TestWriteTriggerResults",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    parameterSetID = cms.string('8b99d66b6c3865c75e460791f721202d'),
    # names should normally be empty. Only extremely old data or
    # has names filled and not empty. If it is not empty, the
    # ParameterSetID is ignored and left default constructed.
    names = cms.vstring(),
    hltStates = cms.vuint32(0, 1, 2, 3),
    moduleIndexes = cms.vuint32(11, 21, 31, 41)
)

process.sender = MPISender(
    upstream = "mpiController",
    instance = 42,
    products = [
        "FEDRawDataCollection_fedRawDataCollectionProducer__*",
        "RawDataBuffer_rawDataBufferProducer__*",
        "edmTriggerResults_triggerResultsProducer__*",
        "triggerTriggerEvent_triggerEventProducer__*"
    ]
)

process.path = cms.Path(
    process.mpiController +
    process.fedRawDataCollectionProducer +
    process.rawDataBufferProducer +
    process.triggerEventProducer +
    process.triggerResultsProducer +
    process.sender
)
