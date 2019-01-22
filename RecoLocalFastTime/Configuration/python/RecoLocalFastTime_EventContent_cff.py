import FWCore.ParameterSet.Config as cms

#FEVT
RecoLocalFastTimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ftlUncalibratedRecHits_*_*',
        'keep *_ftlRecHits_*_*',        
        'keep *_mtdUncalibratedRecHits_*_*',
        'keep *_mtdRecHits_*_*',
        )
)
#RECO content
RecoLocalFastTimeRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ftlRecHits_*_*',        
        )
)
#AOD content
RecoLocalFastTimeAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(    
    )
)
