import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Choose Tracker Geometry
process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.prod = cms.EDAnalyzer("GeoHierarchy",
    fromDDD = cms.bool(True)
)

process.p1 = cms.Path(process.prod)


