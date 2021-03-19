import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMP')

#process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('Geometry.MuonCommonData.testGE0XML_cfi')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometry_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometryDump_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.Geometry=dict()
    process.MessageLogger.GEMNumberingScheme=dict()
    process.MessageLogger.GEMGeometry=dict()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.gemGeometryDump.verbose = True

process.p = cms.Path(process.gemGeometryDump)
