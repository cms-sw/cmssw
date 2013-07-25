import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:/tmp/kkotov/qwe.root')
)

# Little pieces of configuration, taken here and there
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1CSCTFConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

process.csctfanalyzer = cms.EDAnalyzer("CSCTFanalyzer",
         dataTrackProducer = cms.untracked.InputTag("csctfTrackDigis"),
         emulTrackProducer = cms.untracked.InputTag("null"),
         lctProducer       = cms.untracked.InputTag("cscTriggerPrimitiveDigis:MPCSORTED"),
         mbProducer        = cms.untracked.InputTag("csctfunpacker:DT"),
         verbose           = cms.untracked.uint32(1)
)

process.p = cms.Path(process.csctfanalyzer)

