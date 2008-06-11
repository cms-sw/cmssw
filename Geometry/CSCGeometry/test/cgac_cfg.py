import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTestTwo")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.prod = cms.EDFilter("CSCGeometryAsChambers")

process.p1 = cms.Path(process.prod)

