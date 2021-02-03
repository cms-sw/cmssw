import FWCore.ParameterSet.Config as cms

ctppsDiamondRawToDigi = cms.EDProducer("TotemVFATRawToDigi",
    RawToDigi = cms.PSet(
        BC_fraction = cms.untracked.double(0.6),
        BC_min = cms.untracked.uint32(10),
        EC_fraction = cms.untracked.double(0.6),
        EC_min = cms.untracked.uint32(10),
        printErrorSummary = cms.untracked.uint32(0),
        printUnknownFrameSummary = cms.untracked.uint32(0),
        testBCMostFrequent = cms.uint32(2),
        testCRC = cms.uint32(0),
        testECMostFrequent = cms.uint32(0),
        testFootprint = cms.uint32(2),
        testID = cms.uint32(2),
        verbosity = cms.untracked.uint32(0)
    ),
    RawUnpacking = cms.PSet(
        verbosity = cms.untracked.uint32(0)
    ),
    fedIds = cms.vuint32(),
    rawDataTag = cms.InputTag("rawDataCollector"),
    subSystem = cms.string('TimingDiamond')
)
