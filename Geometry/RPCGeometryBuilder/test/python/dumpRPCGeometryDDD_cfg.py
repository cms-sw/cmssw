import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('DUMP',Run3)

process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.RPCGeometryBuilder.rpcGeometry_cfi")
process.load("Geometry.RPCGeometryBuilder.rpcGeometryDump_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('Geometry')
    process.MessageLogger.categories.append('RPCNumberingScheme')
    process.MessageLogger.categories.append('RPCGeometry')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.rpcGeometryDump.verbose = True

process.p = cms.Path(process.rpcGeometryDump)
