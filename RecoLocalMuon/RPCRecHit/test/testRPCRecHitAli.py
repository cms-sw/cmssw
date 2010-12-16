import FWCore.ParameterSet.Config as cms

##process = cms.Process("RPCRecHitAlignment")
process = cms.Process("OWNPARTICLES")


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "MC_3XY_V15::All"
process.GlobalTag.globaltag = 'START3X_V18::All'


process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/data/Run2010B/MuMonitor/RECO/PromptReco-v2/000/149/442/0C5FDCC0-39E6-DF11-A4B2-000423D987E0.root'
        )                           
)

process.load("RecoLocalMuon.RPCRecHit.rpcRecHitAli_cfi")

process.out = cms.OutputModule("PoolOutputModule",
  outputCommands = cms.untracked.vstring('drop *',
        'keep *_dt4DSegments_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcPointProducer_*_*',
        'keep *_rpcRecHits_*_*',
        'keep *_standAloneMuons_*_*',
        'keep *_cosmicMuons_*_*',
        'keep *_globalMuons_*_*'),
 fileName = cms.untracked.string('/tmp/carrillo/outs/output.root')
)
  
process.p = cms.Path(process.rpcRecHitAli)

process.e = cms.EndPath(process.out)


