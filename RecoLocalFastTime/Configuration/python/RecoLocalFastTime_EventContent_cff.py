import FWCore.ParameterSet.Config as cms

#FEVT
RecoLocalFastTimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mtdUncalibratedRecHits_*_*',
        'keep *_mtdRecHits_*_*',
        'keep *_mtdClusters_*_*',
        'keep *_mtdTrackingRecHits_*_*'
        )
)
#RECO content
RecoLocalFastTimeRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mtdRecHits_*_*', 
        'keep *_mtdClusters_*_*',       
        )
)
#AOD content
RecoLocalFastTimeAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
         'keep *_mtdClusters_*_*',
    )
)

