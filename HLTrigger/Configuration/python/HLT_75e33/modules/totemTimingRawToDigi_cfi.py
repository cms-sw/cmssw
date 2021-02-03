import FWCore.ParameterSet.Config as cms

totemTimingRawToDigi = cms.EDProducer("TotemVFATRawToDigi",
    RawToDigi = cms.PSet(
        printErrorSummary = cms.untracked.uint32(0),
        printUnknownFrameSummary = cms.untracked.uint32(0),
        testBCMostFrequent = cms.uint32(0),
        testCRC = cms.uint32(0),
        testECMostFrequent = cms.uint32(0),
        testFootprint = cms.uint32(0),
        testID = cms.uint32(0),
        verbosity = cms.untracked.uint32(0)
    ),
    RawUnpacking = cms.PSet(
        verbosity = cms.untracked.uint32(0)
    ),
    fedIds = cms.vuint32(),
    rawDataTag = cms.InputTag("rawDataCollector"),
    subSystem = cms.string('TotemTiming')
)
