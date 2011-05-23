# /dev/CMSSW_3_11_0/GRun/V18

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTau_selector
streamA_datasetBTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTau_selector.l1tResults = cms.InputTag('')
streamA_datasetBTau_selector.throw      = cms.bool(False)
streamA_datasetBTau_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet10U_v3', 
    'HLT_BTagMu_DiJet20U_Mu5_v3', 
    'HLT_BTagMu_DiJet20U_v3', 
    'HLT_BTagMu_DiJet30U_Mu5_v3', 
    'HLT_BTagMu_DiJet30U_v3', 
    'HLT_DoubleIsoTau15_OneLeg_Trk5_v4', 
    'HLT_DoubleIsoTau15_Trk5_v4', 
    'HLT_SingleIsoTau20_Trk15_MET25_v4', 
    'HLT_SingleIsoTau35_Trk15_MET25_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_CSC', 
    'HLT_Activity_DT', 
    'HLT_Activity_DT_Tuned', 
    'HLT_IsoTrackHB_v2', 
    'HLT_IsoTrackHE_v3', 
    'HLT_L1_BptxXOR_BscMinBiasOR', 
    'HLT_MultiVertex6_v2', 
    'HLT_MultiVertex8_L1ETT60_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_L1MuOpen_AntiBPTX_v2', 
    'HLT_L1Tech_BSC_halo', 
    'HLT_L2Mu0_NoVertex', 
    'HLT_RPCBarrelCosmics', 
    'HLT_TrackerCosmics')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetEGMonitor_selector
streamA_datasetEGMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetEGMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetEGMonitor_selector.throw      = cms.bool(False)
streamA_datasetEGMonitor_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC17', 
    'HLT_Activity_Ecal_SC7', 
    'HLT_DoubleEle4_SW_eeRes_L1R_v2', 
    'HLT_DoubleEle5_SW_Upsilon_L1R_v2', 
    'HLT_Ele10_SW_L1R_v2', 
    'HLT_Ele12_SW_TighterEleId_L1R_v2', 
    'HLT_Ele17_SW_Isol_L1R_v2', 
    'HLT_Ele17_SW_L1R_v2', 
    'HLT_Ele22_SW_L1R_v2', 
    'HLT_L1SingleEG8', 
    'HLT_Photon10_Cleaned_L1R', 
    'HLT_Photon20_Isol_Cleaned_L1R_v1', 
    'HLT_Photon20_NoHE_L1R', 
    'HLT_Photon50_NoHE_L1R')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectron_selector
streamA_datasetElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetElectron_selector.throw      = cms.bool(False)
streamA_datasetElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle17_SW_L1R_v1', 
    'HLT_DoubleEle8_SW_HT70U_L1R_v1', 
    'HLT_Ele10_SW_EleId_HT70U_L1R_v2', 
    'HLT_Ele10_SW_HT100U_L1R_v2', 
    'HLT_Ele10_SW_HT70U_L1R_v2', 
    'HLT_Ele17_SW_TightCaloEleId_Ele8HE_L1R_v2', 
    'HLT_Ele17_SW_TighterEleIdIsol_L1R_v3', 
    'HLT_Ele22_SW_TighterCaloIdIsol_L1R_v2', 
    'HLT_Ele22_SW_TighterEleId_L1R_v3', 
    'HLT_Ele32_SW_TighterEleId_L1R_v2', 
    'HLT_IsoEle12_PFTau15_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise', 
    'HLT_TechTrigHCALNoise')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS', 
    'HLT_HcalPhiSym')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJet_selector
streamA_datasetJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJet_selector.l1tResults = cms.InputTag('')
streamA_datasetJet_selector.throw      = cms.bool(False)
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiJetAve100U_v3', 
    'HLT_DiJetAve140U_v3', 
    'HLT_DiJetAve15U_v3', 
    'HLT_DiJetAve30U_v3', 
    'HLT_DiJetAve50U_v3', 
    'HLT_DiJetAve70U_v3', 
    'HLT_Jet100U_v3', 
    'HLT_Jet140U_v3', 
    'HLT_Jet15U_HcalNoiseFiltered_v3', 
    'HLT_Jet15U_v3', 
    'HLT_Jet180U_v3', 
    'HLT_Jet30U_v3', 
    'HLT_Jet50U_v3', 
    'HLT_Jet70U_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetMETTauMonitor_selector
streamA_datasetJetMETTauMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetMETTauMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetJetMETTauMonitor_selector.throw      = cms.bool(False)
streamA_datasetJetMETTauMonitor_selector.triggerConditions = cms.vstring('HLT_HT50U_v3', 
    'HLT_L1ETT100', 
    'HLT_L1Jet10U', 
    'HLT_L1Jet6U', 
    'HLT_L1MET20', 
    'HLT_MET45_v3', 
    'HLT_QuadJet15U_v3', 
    'HLT_R010U_MR50U')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMETFwd_selector
streamA_datasetMETFwd_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMETFwd_selector.l1tResults = cms.InputTag('')
streamA_datasetMETFwd_selector.throw      = cms.bool(False)
streamA_datasetMETFwd_selector.triggerConditions = cms.vstring('HLT_DoubleJet15U_ForwardBackward_v3', 
    'HLT_DoubleJet25U_ForwardBackward_v3', 
    'HLT_DoubleJet35U_ForwardBackward_v3', 
    'HLT_MET100_v3', 
    'HLT_MET120_v3', 
    'HLT_MET45_DiJet30U_v3', 
    'HLT_MET65_CenJet50U_v3', 
    'HLT_MET80_CenJet50U_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_L1Tech_BSC_HighMultiplicity', 
    'HLT_L1Tech_BSC_halo_forPhysicsBackground', 
    'HLT_L1Tech_BSC_minBias', 
    'HLT_L1Tech_BSC_minBias_OR', 
    'HLT_L1Tech_HCAL_HF', 
    'HLT_L1Tech_RPC_TTU_RBst1_collisions', 
    'HLT_L1_BPTX', 
    'HLT_L1_BPTX_MinusOnly', 
    'HLT_L1_BPTX_PlusOnly', 
    'HLT_MinBiasPixel_SingleTrack', 
    'HLT_PixelTracks_Multiplicity100', 
    'HLT_PixelTracks_Multiplicity70', 
    'HLT_PixelTracks_Multiplicity85', 
    'HLT_Random', 
    'HLT_StoppedHSCP20_v3', 
    'HLT_StoppedHSCP35_v3', 
    'HLT_ZeroBias', 
    'HLT_ZeroBiasPixel_SingleTrack')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMu_selector
streamA_datasetMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMu_selector.l1tResults = cms.InputTag('')
streamA_datasetMu_selector.throw      = cms.bool(False)
streamA_datasetMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_HT50U_v3', 
    'HLT_DoubleMu3_v2', 
    'HLT_DoubleMu5_v1', 
    'HLT_IsoMu11_v4', 
    'HLT_IsoMu13_v4', 
    'HLT_IsoMu15_v4', 
    'HLT_IsoMu17_v4', 
    'HLT_IsoMu9_PFTau15_v2', 
    'HLT_IsoMu9_v4', 
    'HLT_L2DoubleMu20_NoVertex_v1', 
    'HLT_Mu11', 
    'HLT_Mu11_Ele8_v1', 
    'HLT_Mu11_PFTau15_v2', 
    'HLT_Mu13_v1', 
    'HLT_Mu15_v1', 
    'HLT_Mu17_v1', 
    'HLT_Mu19_v1', 
    'HLT_Mu21_v1', 
    'HLT_Mu30_NoVertex_v1', 
    'HLT_Mu3_Ele8_HT70U_v1', 
    'HLT_Mu5_Ele13_v2', 
    'HLT_Mu5_Ele17_v2', 
    'HLT_Mu5_HT100U_v3', 
    'HLT_Mu5_HT70U_v3', 
    'HLT_Mu5_Jet50U_v3', 
    'HLT_Mu5_Jet70U_v3', 
    'HLT_Mu5_MET45_v3', 
    'HLT_Mu7_Photon13_Cleaned_L1R_v1', 
    'HLT_Mu8_Ele8_v1', 
    'HLT_Mu9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuMonitor_selector
streamA_datasetMuMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetMuMonitor_selector.throw      = cms.bool(False)
streamA_datasetMuMonitor_selector.triggerConditions = cms.vstring('HLT_DoubleMu0', 
    'HLT_L1DoubleMuOpen', 
    'HLT_L1Mu20', 
    'HLT_L1Mu7_v1', 
    'HLT_L1MuOpen_DT_v2', 
    'HLT_L1MuOpen_v2', 
    'HLT_L2DoubleMu0', 
    'HLT_L2Mu30_v1', 
    'HLT_L2Mu7_v1', 
    'HLT_Mu0_v2', 
    'HLT_Mu3_v2', 
    'HLT_Mu5', 
    'HLT_Mu7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_DoubleMu0_Quarkonium_v1', 
    'HLT_Mu0_TkMu0_OST_Jpsi_Tight_v3', 
    'HLT_Mu3_TkMu0_OST_Jpsi_Tight_v3', 
    'HLT_Mu3_Track3_Jpsi_v3', 
    'HLT_Mu3_Track5_Jpsi_v3', 
    'HLT_Mu5_L2Mu0', 
    'HLT_Mu5_TkMu0_OST_Jpsi_Tight_v2', 
    'HLT_Mu5_Track0_Jpsi_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_DiJet20U_Meff180U_v3', 
    'HLT_DiJet50U_PT50U_v3', 
    'HLT_EcalOnly_SumEt160_v3', 
    'HLT_ExclDiJet30U_HFAND_v3', 
    'HLT_ExclDiJet30U_HFOR_v3', 
    'HLT_HT100U_v3', 
    'HLT_HT130U_v3', 
    'HLT_HT140U_J30U_Eta3_v3', 
    'HLT_HT150U_Eta3_v3', 
    'HLT_HT150U_v3', 
    'HLT_HT160U_Eta3_v3', 
    'HLT_HT160U_v3', 
    'HLT_HT200U_v3', 
    'HLT_QuadJet20U_v3', 
    'HLT_QuadJet25U_v3', 
    'HLT_R030U_MR100U', 
    'HLT_R033U_MR100U', 
    'HLT_RP025U_MR70U')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoublePhoton17_SingleIsol_L1R_v1', 
    'HLT_DoublePhoton22_L1R_v1', 
    'HLT_DoublePhoton5_CEP_L1R_v3', 
    'HLT_Photon110_NoHE_Cleaned_L1R_v1', 
    'HLT_Photon17_Isol_SC17HE_L1R_v1', 
    'HLT_Photon20_Cleaned_L1R', 
    'HLT_Photon22_SC22HE_L1R_v1', 
    'HLT_Photon30_Cleaned_L1R', 
    'HLT_Photon40_CaloId_Cleaned_L1R_v1', 
    'HLT_Photon40_Isol_Cleaned_L1R_v1', 
    'HLT_Photon50_Cleaned_L1R_v1', 
    'HLT_Photon70_Cleaned_L1R_v1')

