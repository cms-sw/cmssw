import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")

process.source = cms.Source("EmptySource")

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prod = cms.EDAnalyzer("TrackerDigiGeometryAnalyzer")

process.p1 = cms.Path(process.prod)


