# /dev/CMSSW_3_6_2/GRun/V36

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetEG_selector
streamA_datasetEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetEG_selector.l1tResults = cms.InputTag('')
streamA_datasetEG_selector.throw      = cms.bool(False)
streamA_datasetEG_selector.triggerConditions = cms.vstring('HLT_DoublePhoton5_Upsilon_L1R', 
    'HLT_DoublePhoton5_Jpsi_L1R', 
    'HLT_Photon20_Cleaned_L1R', 
    'HLT_DoubleEle10_SW_L1R', 
    'HLT_DoublePhoton15_L1R', 
    'HLT_Ele15_SW_EleId_L1R', 
    'HLT_Ele15_SW_L1R', 
    'HLT_Ele20_SW_L1R', 
    'HLT_DoublePhoton5_CEP_L1R', 
    'HLT_Ele10_SW_EleId_L1R', 
    'HLT_Photon30_Cleaned_L1R', 
    'HLT_Photon50_L1R', 
    'HLT_DoubleEle4_SW_eeRes_L1R', 
    'HLT_Ele15_SW_CaloEleId_L1R', 
    'HLT_Photon50_Cleaned_L1R', 
    'HLT_DoublePhoton20_L1R', 
    'HLT_Ele25_SW_L1R')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_L1Tech_HCAL_HF', 
    'HLT_IsoTrackHB', 
    'HLT_IsoTrackHE', 
    'HLT_L1Tech_RPC_TTU_RBst1_collisions', 
    'HLT_L1Tech_BSC_minBias_OR', 
    'HLT_L1Tech_BSC_HighMultiplicity', 
    'HLT_MinBiasPixel_DoubleIsoTrack5', 
    'HLT_MinBiasPixel_DoubleTrack', 
    'HLT_MinBiasPixel_SingleTrack', 
    'HLT_ZeroBiasPixel_SingleTrack', 
    'HLT_L1Tech_BSC_minBias', 
    'HLT_StoppedHSCP', 
    'HLT_L1_BPTX_PlusOnly', 
    'HLT_L1_BPTX_MinusOnly', 
    'HLT_L1_BPTX', 
    'HLT_ZeroBias', 
    'HLT_L1Tech_BSC_halo_forPhysicsBackground', 
    'HLT_PixelTracks_Multiplicity85', 
    'HLT_MultiVertex6', 
    'HLT_MultiVertex8_L1ETT60', 
    'HLT_PixelTracks_Multiplicity70')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTau_selector
streamA_datasetBTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTau_selector.l1tResults = cms.InputTag('')
streamA_datasetBTau_selector.throw      = cms.bool(False)
streamA_datasetBTau_selector.triggerConditions = cms.vstring('HLT_BTagIP_Jet50U', 
    'HLT_DoubleLooseIsoTau15', 
    'HLT_SingleLooseIsoTau20', 
    'HLT_BTagMu_Jet10U', 
    'HLT_SingleIsoTau20_Trk5', 
    'HLT_SingleLooseIsoTau25_Trk5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetMET_selector
streamA_datasetJetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetJetMET_selector.throw      = cms.bool(False)
streamA_datasetJetMET_selector.triggerConditions = cms.vstring('HLT_HT100U', 
    'HLT_MET100', 
    'HLT_MET45', 
    'HLT_QuadJet15U', 
    'HLT_DiJetAve30U', 
    'HLT_DiJetAve15U', 
    'HLT_FwdJet20U', 
    'HLT_Jet50U', 
    'HLT_Jet30U', 
    'HLT_Jet15U', 
    'HLT_DoubleJet15U_ForwardBackward', 
    'HLT_Jet15U_HcalNoiseFiltered', 
    'HLT_DiJetAve50U', 
    'HLT_Jet100U', 
    'HLT_Jet70U', 
    'HLT_EcalOnly_SumEt160', 
    'HLT_L1ETT100')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetEGMonitor_selector
streamA_datasetEGMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetEGMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetEGMonitor_selector.throw      = cms.bool(False)
streamA_datasetEGMonitor_selector.triggerConditions = cms.vstring('HLT_L1SingleEG2', 
    'HLT_DoublePhoton10_L1R', 
    'HLT_Photon10_Cleaned_L1R', 
    'HLT_L1DoubleEG5', 
    'HLT_Ele15_SiStrip_L1R', 
    'HLT_Ele15_LW_L1R', 
    'HLT_L1SingleEG8', 
    'HLT_L1SingleEG5', 
    'HLT_SelectEcalSpikes_L1R', 
    'HLT_SelectEcalSpikesHighEt_L1R', 
    'HLT_DoublePhoton5_L1R', 
    'HLT_Photon15_Cleaned_L1R', 
    'HLT_Activity_Ecal_SC7', 
    'HLT_Activity_Ecal_SC17', 
    'HLT_Ele20_SiStrip_L1R')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetMETTauMonitor_selector
streamA_datasetJetMETTauMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetMETTauMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetJetMETTauMonitor_selector.throw      = cms.bool(False)
streamA_datasetJetMETTauMonitor_selector.triggerConditions = cms.vstring('HLT_L1Jet10U_NoBPTX', 
    'HLT_L1Jet6U', 
    'HLT_L1Jet6U_NoBPTX', 
    'HLT_L1SingleCenJet', 
    'HLT_L1SingleForJet', 
    'HLT_L1SingleTauJet', 
    'HLT_L1MET20', 
    'HLT_L1Jet10U')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_TechTrigHCALNoise', 
    'HLT_GlobalRunHPDNoise')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetRandomTriggers_selector
streamA_datasetRandomTriggers_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetRandomTriggers_selector.l1tResults = cms.InputTag('')
streamA_datasetRandomTriggers_selector.throw      = cms.bool(False)
streamA_datasetRandomTriggers_selector.triggerConditions = cms.vstring('HLT_Random')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS', 
    'HLT_HcalPhiSym')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_CSCBeamHaloOverlapRing2', 
    'HLT_CSCBeamHaloOverlapRing1', 
    'HLT_CSCBeamHalo', 
    'HLT_L1Tech_BSC_halo', 
    'HLT_L1MuOpen_AntiBPTX', 
    'HLT_L2Mu0_NoVertex', 
    'HLT_TrackerCosmics', 
    'HLT_RPCBarrelCosmics', 
    'HLT_CSCBeamHaloRing2or3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMu_selector
streamA_datasetMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMu_selector.l1tResults = cms.InputTag('')
streamA_datasetMu_selector.throw      = cms.bool(False)
streamA_datasetMu_selector.triggerConditions = cms.vstring('HLT_L1Mu14_L1ETM30', 
    'HLT_L1Mu14_L1SingleJet6U', 
    'HLT_L1Mu14_L1SingleEG10', 
    'HLT_L1Mu20', 
    'HLT_DoubleMu3', 
    'HLT_Mu3', 
    'HLT_Mu5', 
    'HLT_Mu9', 
    'HLT_L2Mu9', 
    'HLT_L2Mu11', 
    'HLT_L1Mu30', 
    'HLT_Mu7', 
    'HLT_L2Mu15')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Mu3_Track0_Jpsi', 
    'HLT_Mu5_Track0_Jpsi', 
    'HLT_Mu3_L2Mu0', 
    'HLT_Mu5_L2Mu0', 
    'HLT_L2DoubleMu0', 
    'HLT_DoubleMu0', 
    'HLT_L1DoubleMuOpen_Tight', 
    'HLT_Mu0_TkMu0_Jpsi', 
    'HLT_Mu0_TkMu0_Jpsi_NoCharge', 
    'HLT_Mu3_TkMu0_Jpsi', 
    'HLT_Mu3_TkMu0_Jpsi_NoCharge', 
    'HLT_Mu5_TkMu0_Jpsi', 
    'HLT_Mu5_TkMu0_Jpsi_NoCharge')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuMonitor_selector
streamA_datasetMuMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetMuMonitor_selector.throw      = cms.bool(False)
streamA_datasetMuMonitor_selector.triggerConditions = cms.vstring('HLT_Mu0_L1MuOpen', 
    'HLT_Mu0_Track0_Jpsi', 
    'HLT_Mu3_L1MuOpen', 
    'HLT_Mu5_L1MuOpen', 
    'HLT_L1DoubleMuOpen', 
    'HLT_L1Mu', 
    'HLT_L1MuOpen', 
    'HLT_Mu0_L2Mu0', 
    'HLT_L2Mu0', 
    'HLT_L2Mu3', 
    'HLT_L2Mu5', 
    'HLT_L1MuOpen_DT')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_DT', 
    'HLT_Activity_DT_Tuned', 
    'HLT_Activity_PixelClusters', 
    'HLT_Activity_L1A', 
    'HLT_Activity_CSC', 
    'HLT_L1_BptxXOR_BscMinBiasOR')

