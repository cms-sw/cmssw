import FWCore.ParameterSet.Config as cms

#FEVT
RecoLocalFastTimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ftlUncalibratedRecHits_*_*',
        'keep *_ftlRecHits_*_*',        
        'keep *_mtdUncalibratedRecHits_*_*',
        'keep *_mtdRecHits_*_*',
        'keep *_mtdClusters_*_*',
        'keep *_mtdTrackingRecHits_*_*'
        )
)
#RECO content
RecoLocalFastTimeRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ftlRecHits_*_*',
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

from Configuration.Eras.Modifier_phase2_timing_layer_tile_cff import phase2_timing_layer_tile
from Configuration.Eras.Modifier_phase2_timing_layer_bar_cff import phase2_timing_layer_bar

(phase2_timing_layer_tile | phase2_timing_layer_bar).toModify( 
    RecoLocalFastTimeFEVT.outputCommands, 
    func=lambda outputCommands: outputCommands.extend(['drop *_ftlUncalibratedRecHits_*_*',
                                                       'drop *_ftlRecHits_*_*'])
)

(phase2_timing_layer_tile | phase2_timing_layer_bar).toModify( 
    RecoLocalFastTimeRECO.outputCommands, 
    func=lambda outputCommands: outputCommands.extend(['drop *_ftlRecHits_*_*']) 
)
