import FWCore.ParameterSet.Config as cms

# Name:   RecoMET_EventContent.cff
# Author: R.Cavanaugh
# Date:   05.11.2006
# Notes:
# Modification: F. Blekman (added CaloMETSignif), JP Chou (added HcalNoise)
# Full Event content 
RecoMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*', 
                                           'keep recoMETs_*_*_*',
                                           'keep recoPFMETs_*_*_*',
                                           #'keep recoCaloMETSignifs_*_*_*'
                                           )
    )
RecoGenMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
    )
RecoHcalNoiseFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoHcalNoiseRBXs_*_*_*',
                                           'keep HcalNoiseSummary_*_*_*'
                                           )
    )
# RECO content
RecoMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*',
                                           'keep recoMETs_*_*_*',
                                           'keep recoPFMETs_*_*_*',
                                           #'keep recoCaloMETSignifs_*_*_*'
                                           )
    )
RecoGenMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
    )
RecoHcalNoiseRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoHcalNoiseRBXs_*_*_*',
                                           'keep HcalNoiseSummary_*_*_*'
                                           )
    )
# AOD content
RecoMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*',
                                           'keep recoMETs_*_*_*',
                                           'keep recoPFMETs_*_*_*',
                                           #'keep recoCaloMETSignifs_*_*_*'
                                           )
    )
RecoGenMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
    )
RecoHcalNoiseAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop recoHcalNoiseRBXs_*_*_*',
                                           'keep HcalNoiseSummary_*_*_*'
                                           )
    )
