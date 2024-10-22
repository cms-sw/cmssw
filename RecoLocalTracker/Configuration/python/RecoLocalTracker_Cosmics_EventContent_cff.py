import FWCore.ParameterSet.Config as cms

# AOD content
RecoLocalTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

# RECO content
RecoLocalTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep PixelDigiedmDetSetVector_siPixelDigis_*_*', 
        'keep *_siStripDigis_*_*', 
        'keep *_siStripZeroSuppression_*_*', 
        'keep *_siPixelClusters_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelRecHits_*_*', 
        'keep *_siStripRecHits_*_*', 
        'keep *_siStripMatchedRecHits_*_*')
)
RecoLocalTrackerRECO.outputCommands.extend(RecoLocalTrackerAOD.outputCommands)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(RecoLocalTrackerRECO, outputCommands = RecoLocalTrackerRECO.outputCommands + ['keep *_siPhase2Clusters_*_*','keep *_siPhase2RecHits_*_*'] )

# FEVT content
RecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoLocalTrackerFEVT.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)

phase2_tracker.toModify(RecoLocalTrackerFEVT, outputCommands = RecoLocalTrackerFEVT.outputCommands + ['keep *_siPhase2Clusters_*_*','keep *_siPhase2RecHits_*_*'] )
