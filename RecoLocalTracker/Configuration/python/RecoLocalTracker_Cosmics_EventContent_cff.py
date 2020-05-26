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

# FEVT content
RecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoLocalTrackerFEVT.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
