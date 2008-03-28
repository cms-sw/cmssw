# The following comments couldn't be translated into the new config version:

#Tracks

#Tracks

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*')
)
#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*')
)
#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTracks_rsWithMaterialTracks_*_*')
)

