import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load('Configuration.Geometry.GeometryExtended_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

process.GlobalTag.globaltag = 'MC_31X_V8::All'
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.out = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("DTGeometryAnalyzer")

process.p1 = cms.Path(process.prod)


