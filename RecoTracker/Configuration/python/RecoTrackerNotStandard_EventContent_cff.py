# The following comments couldn't be translated into the new config version:

#Tracks

#Tracks

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTrackerNotStandardFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfNoOverlaps_*_*', 'keep *_ctfPixelLess_*_*', 'keep *_ctfCombinedSeeds_*_*')
)
#RECO content
RecoTrackerNotStandardRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfNoOverLaps_*_*', 'keep *_ctfPixelLess_*_*', 'keep *_ctfCombinedSeeds_*_*')
)
#AOD content
RecoTrackerNotStandardAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfNoOverLaps_*_*', 'keep recoTracks_ctfPixelLess_*_*', 'keep recoTracks_ctfCombinedSeeds_*_*')
)

