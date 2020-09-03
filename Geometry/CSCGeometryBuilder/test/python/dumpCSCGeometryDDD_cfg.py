import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('DUMP',Run3)

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.CSCGeometryBuilder.cscGeometry_cfi")
process.load("Geometry.CSCGeometryBuilder.cscGeometryDump_cfi")

process.CSCGeometryESModule.applyAlignment = False
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('Geometry')
    process.MessageLogger.categories.append('CSCNumberingScheme')
    process.MessageLogger.categories.append('CSCGeometry')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.cscGeometryDump.verbose = True

process.p = cms.Path(process.cscGeometryDump)
