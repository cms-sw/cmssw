# The following comments couldn't be translated into the new config version:

#Tracks

#Tracks

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracksTIF_*_*', 
        'keep *_rsWithMaterialTracksTIF_*_*')
)
#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracksTIF_*_*', 
        'keep *_rsWithMaterialTracksTIF_*_*')
)
#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksTIF_*_*', 
        'keep recoTracks_rsWithMaterialTracksTIF_*_*')
)

