import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCRecHitAlignment")


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START3X_V18::All'


process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/data/Run2010B/MuMonitor/RECO/PromptReco-v2/000/149/442/0C5FDCC0-39E6-DF11-A4B2-000423D987E0.root'
        )                           
)

process.load("RecoLocalMuon.RPCRecHit.rpcRecHitAli_cfi")

process.out = cms.OutputModule("PoolOutputModule",
 fileName = cms.untracked.string('/tmp/carrillo/outs/output.root')
)
  
process.p = cms.Path(process.rpcRecHitAli)

process.e = cms.EndPath(process.out)


