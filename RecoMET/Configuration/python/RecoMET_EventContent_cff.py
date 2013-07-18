import FWCore.ParameterSet.Config as cms
# $Id: RecoMET_EventContent_cff.py,v 1.15 2012/09/04 21:18:33 sakuma Exp $

##______________________________________________________ Full Event content __||
RecoMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_met_*_*',
                                           'keep recoCaloMETs_metNoHF_*_*',
                                           'keep recoCaloMETs_metHO_*_*',
                                           'keep recoCaloMETs_corMetGlobalMuons_*_*',
                                           'keep recoMETs_tcMet_*_*',
                                           'keep recoMETs_tcMetWithPFclusters_*_*',                                           
                                           'keep recoPFMETs_pfMet_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_muonTCMETValueMapProducer_*_*',                                           
                                           'keep recoHcalNoiseRBXs_hcalnoise_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*',
                                           'keep *HaloData_*_*_*',
                                           'keep *BeamHaloSummary_BeamHaloSummary_*_*'
                                           )
    )

RecoGenMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
    )

RecoHcalNoiseFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoHcalNoiseRBXs_hcalnoise_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*'
                                           )
    )

##____________________________________________________________ RECO content __||
RecoMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_met_*_*',
                                           'keep recoCaloMETs_metNoHF_*_*',
                                           'keep recoCaloMETs_metHO_*_*',
                                           'keep recoCaloMETs_corMetGlobalMuons_*_*',
                                           'keep recoMETs_tcMet_*_*',    
                                           'keep recoMETs_tcMetWithPFclusters_*_*',                                                                                  
                                           'keep recoPFMETs_pfMet_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_muonTCMETValueMapProducer_*_*',                                           
                                           'keep recoHcalNoiseRBXs_hcalnoise_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*',
                                           'keep *HaloData_*_*_*',
                                           'keep *BeamHaloSummary_BeamHaloSummary_*_*'
                                           )
    )

RecoGenMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
    )

RecoHcalNoiseRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoHcalNoiseRBXs_hcalnoise_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*'
                                           )
    )

##_____________________________________________________________ AOD content __||
RecoMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_met_*_*',
                                           'keep recoCaloMETs_metNoHF_*_*',
                                           'keep recoCaloMETs_metHO_*_*',
                                           'keep recoCaloMETs_corMetGlobalMuons_*_*',
                                           'keep recoMETs_tcMet_*_*',  
                                           'keep recoMETs_tcMetWithPFclusters_*_*',                                                                                                                           
                                           'keep recoPFMETs_pfMet_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*',
                                           'keep recoMuonMETCorrectionDataedmValueMap_muonTCMETValueMapProducer_*_*',                                           
                                           'drop recoHcalNoiseRBXs_*_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*',
                                           'keep *GlobalHaloData_*_*_*',
                                           'keep *BeamHaloSummary_BeamHaloSummary_*_*'
                                           )
    )

RecoGenMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
    )

RecoHcalNoiseAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop recoHcalNoiseRBXs_hcalnoise_*_*',
                                           'keep HcalNoiseSummary_hcalnoise_*_*'
                                           )
    )

##____________________________________________________________________________||
