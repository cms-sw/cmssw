import FWCore.ParameterSet.Config as cms

#FEVT
RecoLocalFastTimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ftlUncalibratedRecHits_*_*',
        'keep *_ftlRecHits_*_*',        
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

from Configuration.Eras.Modifier_phase2_timing_layer_new_cff import phase2_timing_layer_new
phase2_timing_layer_new.toModify( RecoLocalFastTimeFEVT.outputCommands, func=lambda outputCommands: outputCommands.extend(['drop *_ftlUncalibratedRecHits_*_*','keep *_mtdUncalibratedRecHits_*_*']) )
