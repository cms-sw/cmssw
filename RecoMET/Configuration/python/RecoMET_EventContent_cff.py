import FWCore.ParameterSet.Config as cms

# Name:   RecoMET_EventContent.cff
# Author: R.Cavanaugh
# Date:   05.11.2006
# Notes:  
# Full Event content 
RecoMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*')
)
RecoGenMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
)
# RECO content
RecoMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*')
)
RecoGenMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
)
# AOD content
RecoMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*')
)
RecoGenMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
)

