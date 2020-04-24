import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
                                                     fromDDD = cms.bool(True))

process.prod = cms.EDAnalyzer("GeometricDetAnalyzer")

process.p1 = cms.Path(process.prod)

