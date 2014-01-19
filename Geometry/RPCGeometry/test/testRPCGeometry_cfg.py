import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

# process.load('Configuration.Geometry.GeometryExtended_cff')
# process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
# process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

# from Configuration.AlCa.autoCond import autoCond
# process.GlobalTag.globaltag = autoCond['mc']

process.load("Configuration.Geometry.GeometryExtended2023RPCEtaUpscope_cff")
process.load('Configuration.Geometry.GeometryExtended2023RPCUpscopeReco_cff')
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

process.test1 = cms.EDAnalyzer("RPCGEO")
process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.p = cms.Path(process.test1+process.test2)

