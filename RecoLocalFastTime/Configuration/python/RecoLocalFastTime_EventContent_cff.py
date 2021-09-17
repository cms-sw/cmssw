import FWCore.ParameterSet.Config as cms

#AOD content
RecoLocalFastTimeAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

#RECO content
RecoLocalFastTimeRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mtdRecHits_*_*',
        'keep *_mtdClusters_*_*',
    )
)
RecoLocalFastTimeRECO.outputCommands.extend(RecoLocalFastTimeAOD.outputCommands)

#FEVT
RecoLocalFastTimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mtdUncalibratedRecHits_*_*',
        'keep *_mtdTrackingRecHits_*_*',
    )
)
RecoLocalFastTimeFEVT.outputCommands.extend(RecoLocalFastTimeRECO.outputCommands)
