import FWCore.ParameterSet.Config as cms

# Pick branches you want to keep
EXONoBPTXSkim_EventContent = cms.PSet(
     outputCommands = cms.untracked.vstring(
                     'drop *',
                     'keep *_BeamHaloSummary_*_*',
                     'keep recoCaloJets_ak4CaloJets_*_*',
                     'keep *_hcalnoise_*_*',
                     'keep *_towerMaker_*_*',
                     'keep *_hbhereco_*_*',
                     'keep *_hfreco_*_*',
                     'keep *_hfprereco_*_*',
                     'keep *_cscSegments_*_*',
                     'keep *_dt4DSegments_*_*',
                     'keep *_rpcRecHits_*_*',
                     'keep recoMuons_muons_*_*',
                     'keep *_muons_dt_*',
                     'keep *_muons_csc_*',
                     'keep *_muons_combined_*',
                     'keep recoMuons_muonsFromCosmics1Leg_*_*',
                     'keep recoMuons_muonsFromCosmics_*_*',
                     'keep *_standAloneMuons_*_*',
                     'keep *_displacedStandAloneMuons_*_*',
                     'keep recoTracks_generalTracks_*_*',
                     'keep recoVertexs_offlinePrimaryVertices_*_*',
                     'keep *_TriggerResults_*_*',
     )
)
