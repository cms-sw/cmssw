import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMP')

process.load('Geometry.MuonCommonData.testGEMXML_cfi')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometry_cff")
process.load("Geometry.GEMGeometryBuilder.me0Geometry_cff")
process.load("Geometry.GEMGeometryBuilder.me0GeometryDump_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('Geometry')
    process.MessageLogger.categories.append('ME0NumberingScheme')
    process.MessageLogger.categories.append('ME0Geometry')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.gemGeometryDump.verbose = True

process.p = cms.Path(process.me0GeometryDump)
