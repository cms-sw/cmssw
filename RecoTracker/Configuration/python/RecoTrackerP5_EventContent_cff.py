# The following comments couldn't be translated into the new config version:

#Tracks

#Tracks

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracksP5_*_*', 'keep *_rsWithMaterialTracksP5_*_*')
)
#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracksP5_*_*', 'keep *_rsWithMaterialTracksP5_*_*')
)
#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksP5_*_*', 'keep recoTracks_rsWithMaterialTracksP5_*_*')
)

