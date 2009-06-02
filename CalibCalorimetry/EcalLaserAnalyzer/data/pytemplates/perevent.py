import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTBIS")
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

process.PerEvtLaserAnalyzer = cms.EDFilter("EcalPerEvtLaserAnalyzer",
    resDir = cms.untracked.string('.'),
    eventHeaderCollection = cms.string(''),
    eventHeaderProducer = cms.string('ecalEBunpacker'),
    digiCollection = cms.string('CCCC'),
    digiPNCollection = cms.string(''),
    fedID = cms.untracked.int32(FFFF),
    digiProducer = cms.string('ecalEBunpacker'),
    useOneABPerCrys = cms.untracked.bool(True),
    ecalPart = cms.untracked.string('PPPP'),
    tower = cms.untracked.uint32(1),
    channel = cms.untracked.uint32(1),
    refAlphaBeta = cms.untracked.string('AB.root')
)

process.p = cms.Path(process.ecalEBunpacker*process.PerEvtLaserAnalyzer)


