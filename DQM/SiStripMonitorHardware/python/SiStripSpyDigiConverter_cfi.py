import FWCore.ParameterSet.Config as cms

SiStripSpyDigiConverter = cms.EDProducer(
    "SiStripSpyDigiConverterModule",
    InputProductLabel = cms.InputTag('SiStripSpyUnpacker','ScopeRawDigis'),
    StorePayloadDigis = cms.bool(False),
    StoreReorderedDigis = cms.bool(False),
    StoreModuleDigis = cms.bool(True),
    StoreAPVAddress = cms.bool(True),
    DiscardDigisWithWrongAPVAddress = cms.bool(True),
    MinDigiRange = cms.uint32(100),
    MaxDigiRange = cms.uint32(1024),
    MinZeroLight = cms.uint32(0),
    MaxZeroLight = cms.uint32(1024),
    MinTickHeight = cms.uint32(0),
    MaxTickHeight = cms.uint32(1024),
    ExpectedPositionOfFirstHeaderBit = cms.uint32(6)
    )
