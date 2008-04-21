# The following comments couldn't be translated into the new config version:

#Clusters

#Si Pixel hits

#Si Strip hits

#Clusters

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_siPixelClusters_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelRecHits_*_*', 
        'keep *_siStripRecHits_*_*', 
        'keep *_siStripMatchedRecHits_*_*')
)
#RECO content
RecoLocalTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_siPixelClusters_*_*', 
        'keep *_siStripClusters_*_*')
)
#AOD content
RecoLocalTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

