import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process('DUMP',Run3_dd4hep)

#process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.RPCGeometryBuilder.rpcGeometry_cfi")
process.load("Geometry.RPCGeometryBuilder.rpcGeometryDump_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.Geometry=dict()
    process.MessageLogger.RPCNumberingScheme=dict()
    process.MessageLogger.RPCGeometry=dict()
    process.MessageLogger.RPCGeometryBuilder=dict()
    process.MessageLogger.RPCGeometryParsFromDD=dict()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.rpcGeometryDump.verbose = True

process.p = cms.Path(process.rpcGeometryDump)
