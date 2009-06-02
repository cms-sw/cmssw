import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTMON")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("LmfSource",
    orderedRead = cms.bool(True),
    verbosity = cms.untracked.int32(0),
    fileNames = cms.vstring('input.lmf'),
    preScale = cms.uint32(1)
)

process.ABAnalyzer = cms.EDFilter("EcalABAnalyzer",
    resDir = cms.untracked.string('.'),
    eventHeaderProducer = cms.string('ecalEBunpacker'),
    eventHeaderCollection = cms.string(''),
    digiCollection = cms.string('CCCC'),
    digiPNCollection = cms.string(''),
    digiProducer = cms.string('ecalEBunpacker'),
    fedID = cms.untracked.int32(FFFF)
)

process.p = cms.Path(process.ecalEBunpacker*process.ABAnalyzer)


