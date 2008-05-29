# The following comments couldn't be translated into the new config version:

#Clusters

#Clusters

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_siPixelClusters_*_*', 
        'keep *_siStripClusters_*_*')
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

