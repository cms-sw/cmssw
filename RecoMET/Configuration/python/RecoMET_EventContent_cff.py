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
                                           'keep recoMuonMETCorrectionDataedmValueMap_*_*_*',
                                           'keep recoHcalNoiseRBXs_*_*_*',
                                           'keep HcalNoiseSummary_*_*_*',
                                           'keep *HaloData_*_*_*',
                                           'keep *BeamHaloSummary_*_*_*'
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
                                           'keep recoMuonMETCorrectionDataedmValueMap_*_*_*',
                                           'keep recoHcalNoiseRBXs_*_*_*',
                                           'keep HcalNoiseSummary_*_*_*',
                                           'keep *HaloData_*_*_*',
                                           'keep *BeamHaloSummary_*_*_*'
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
                                           'drop recoCaloMETs_metOptHO_*_*',
                                           'drop recoCaloMETs_metOpt_*_*',
                                           'drop recoCaloMETs_metOptNoHFHO_*_*',
                                           'drop recoCaloMETs_metNoHFHO_*_*',
                                           'drop recoCaloMETs_metOptNoHF_*_*',
                                           'keep recoMETs_*_*_*',
                                           'keep recoPFMETs_*_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_*_*_*',
                                           'drop recoHcalNoiseRBXs_*_*_*',

                                           'keep HcalNoiseSummary_*_*_*',
                                           'keep *GlobalHaloData_*_*_*',
                                           'keep *BeamHaloSummary_*_*_*'
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
