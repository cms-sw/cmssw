import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
# process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
# process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
# process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.Geometry.GeometryExtended2023_cff")
process.load("Configuration.Geometry.GeometryExtended2023Reco_cff")
# process.load("Configuration.Geometry.GeometryExtended2023RPCUpscope_2p4_192_cff")
# process.load('Configuration.Geometry.GeometryExtended2023RPCUpscopeReco_2p4_192_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
#process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.demo = cms.EDAnalyzer("RPCGEO")

process.p = cms.Path(process.demo)

