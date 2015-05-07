import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

# process.load("Configuration.Geometry.GeometryExtended2015Reco_cff")
# process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
process.load("Configuration.Geometry.GeometryExtended2015Reco_RPC4RE11_cff")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source        = cms.Source("EmptySource")
process.MessageLogger = cms.Service("MessageLogger")
process.demo          = cms.EDAnalyzer("RPCGEO")
process.p             = cms.Path(process.demo)

