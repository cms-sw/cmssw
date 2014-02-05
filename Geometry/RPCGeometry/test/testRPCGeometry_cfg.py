import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
# process.load('Configuration.Geometry.GeometryExtended2023_cff')
# process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
# process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

# process.load('Configuration.Geometry.GeometryExtended2023HGCalMuon_cff')
# process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
# process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# process.load('Configuration.Geometry.GeometryExtended_cff')
# process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
# process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

# from Configuration.AlCa.autoCond import autoCond
# process.GlobalTag.globaltag = autoCond['mc']


process.load('Configuration.Geometry.GeometryExtended2023_cff') 
process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')

# process.load('Configuration.Geometry.GeometryExtended2023RPCUpscopeReco_2p4_192_cff')
# process.load("Configuration.Geometry.GeometryExtended2023RPCUpscope_2p4_192_cff")
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
# process.MessageLogger = cms.Service("MessageLogger",                                   
#     debugModules = cms.untracked.vstring('*'),
#     cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
#     # cout = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
#     destinations = cms.untracked.vstring('cout')
# )
process.test1 = cms.EDAnalyzer("RPCGEO")
process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.p = cms.Path(process.test1+process.test2)

