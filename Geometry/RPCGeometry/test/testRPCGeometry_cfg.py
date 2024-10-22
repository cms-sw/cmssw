import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load('Configuration.Geometry.GeometryExtended_cff')
process.load('Configuration.Geometry.GeometryExtendedReco_cff')

process.load('FWCore.MessageLogger.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.load('FWCore.MessageLogger.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.RPCGeometry=dict()

process.test1 = cms.EDAnalyzer("RPCGEO")
process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.p = cms.Path(process.test1+process.test2)

