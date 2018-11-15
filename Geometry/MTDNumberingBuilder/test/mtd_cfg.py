import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryExtended2023D24_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MTDGeometricTimingDetESModule = cms.ESProducer("MTDGeometricTimingDetESModule",
                                                     fromDDD = cms.bool(True))

process.prod = cms.EDAnalyzer("GeometricTimingDetAnalyzer")

process.p1 = cms.Path(process.prod)

