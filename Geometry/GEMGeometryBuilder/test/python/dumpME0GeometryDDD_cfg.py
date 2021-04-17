import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMP')

process.load('Geometry.MuonCommonData.testGEMXML_cfi')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometry_cff")
process.load("Geometry.GEMGeometryBuilder.me0Geometry_cff")
process.load("Geometry.GEMGeometryBuilder.me0GeometryDump_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.Geometry=dict()
    process.MessageLogger.ME0NumberingScheme=dict()
    process.MessageLogger.ME0Geometry=dict()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.me0GeometryDump.verbose = True

process.p = cms.Path(process.me0GeometryDump)
