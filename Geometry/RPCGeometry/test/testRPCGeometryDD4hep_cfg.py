import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("Demo", Run3_dd4hep)

process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

process.load('FWCore.MessageLogger.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.RPCGeometry=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.test1 = cms.EDAnalyzer("RPCGEO")
process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.p = cms.Path(process.test1+process.test2)

