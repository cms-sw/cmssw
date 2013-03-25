import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load('Configuration.Geometry.GeometryExtendedPostLS1Reco_cff')
#process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
#process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
#process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.demo = cms.EDAnalyzer("RPCGEO_POSTLS1")

process.p = cms.Path(process.demo)

