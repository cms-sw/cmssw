# /dev/CMSSW_4_2_0/GRun/V16

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v2', 
    'HLT_BeamGas_BSC_v2', 
    'HLT_BeamGas_HF_v3', 
    'HLT_IsoTrackHB_v2', 
    'HLT_IsoTrackHE_v3', 
    'HLT_L1SingleEG12_v1', 
    'HLT_L1SingleEG5_v1', 
    'HLT_L1SingleJet16_v1', 
    'HLT_L1SingleJet36_v1', 
    'HLT_L1SingleMuOpen_DT_v1', 
    'HLT_L1SingleMuOpen_v1', 
    'HLT_L1_Interbunch_BSC_v1', 
    'HLT_L1_PreCollisions_v1', 
    'HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v2', 
    'HLT_Photon20_EBOnly_NoSpikeFilter_v1', 
    'HLT_Photon20_NoSpikeFilter_v1', 
    'HLT_Spike20_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v2', 
    'HLT_L1SingleMuOpen_AntiBPTX_v1', 
    'HLT_L1Tech_BSC_halo_v3', 
    'HLT_L1TrackerCosmics_v2', 
    'HLT_L3MuonsCosmicTracking_v2', 
    'HLT_RegionalCosmicTracking_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_v3', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele15_HFL_v3', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele8_CaloIdL_CaloIsoVL_v3', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v3', 
    'HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v3', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v3', 
    'HLT_Ele32_CaloIdL_CaloIsoVL_SC17_v3', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v3', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v3', 
    'HLT_Ele8_CaloIdL_TrkIdVL_v3', 
    'HLT_Ele8_v3', 
    'HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v3', 
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_v4', 
    'HLT_DoubleMu4_Acoplanarity03_v2', 
    'HLT_DoubleMu6_v2', 
    'HLT_DoubleMu7_v2', 
    'HLT_L1DoubleMu0_v1', 
    'HLT_L2DoubleMu0_v3', 
    'HLT_L2DoubleMu23_NoVertex_v2', 
    'HLT_Mu8_Jet40_v4', 
    'HLT_TripleMu5_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_DoubleEle8_CaloIdL_TrkIdVL_HT150_v1', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_HT150_v1', 
    'HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT200_v4', 
    'HLT_Ele10_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT200_v4', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta2_v1', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta3_v1', 
    'HLT_Ele15_CaloIdVT_TrkIdT_Jet35_Jet25_Deta2_v1', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralDiJet30_v3', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_BTagIP_v2', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_v3', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralTriJet30_v3', 
    'HLT_HT200_Ele5_CaloIdVL_TrkIdVL_CaloIsoVL_TrkIsoVL_PFMHT35_v2', 
    'HLT_HT250_Ele5_CaloIdVL_TrkIdVL_CaloIsoVL_TrkIsoVL_PFMHT35_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetForwardTriggers_selector
streamA_datasetForwardTriggers_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetForwardTriggers_selector.l1tResults = cms.InputTag('')
streamA_datasetForwardTriggers_selector.throw      = cms.bool(False)
streamA_datasetForwardTriggers_selector.triggerConditions = cms.vstring('HLT_L1Tech_CASTOR_HaloMuon_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHT_selector
streamA_datasetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHT_selector.throw      = cms.bool(False)
streamA_datasetHT_selector.triggerConditions = cms.vstring('HLT_HT150_AlphaT0p60_v2', 
    'HLT_HT150_AlphaT0p70_v2', 
    'HLT_HT150_v3', 
    'HLT_HT200_AlphaT0p60_v2', 
    'HLT_HT200_AlphaT0p65_v2', 
    'HLT_HT200_v3', 
    'HLT_HT250_AlphaT0p55_v2', 
    'HLT_HT250_AlphaT0p62_v2', 
    'HLT_HT250_DoubleDisplacedJet60_v2', 
    'HLT_HT250_MHT60_v3', 
    'HLT_HT250_v3', 
    'HLT_HT300_AlphaT0p52_v2', 
    'HLT_HT300_AlphaT0p54_v2', 
    'HLT_HT300_MHT75_v4', 
    'HLT_HT300_v4', 
    'HLT_HT350_AlphaT0p51_v2', 
    'HLT_HT350_AlphaT0p53_v2', 
    'HLT_HT350_v3', 
    'HLT_HT400_AlphaT0p51_v2', 
    'HLT_HT400_v3', 
    'HLT_HT450_v3', 
    'HLT_HT500_v3', 
    'HLT_HT550_v3', 
    'HLT_MR100_v2', 
    'HLT_Meff440_v3', 
    'HLT_Meff520_v3', 
    'HLT_Meff640_v3', 
    'HLT_R032_MR100_v2', 
    'HLT_R032_v2', 
    'HLT_R035_MR100_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v2', 
    'HLT_L1Tech_HBHEHO_totalOR_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v3', 
    'HLT_HcalPhiSym_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJet_selector
streamA_datasetJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJet_selector.l1tResults = cms.InputTag('')
streamA_datasetJet_selector.throw      = cms.bool(False)
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiJetAve110_v1', 
    'HLT_DiJetAve150_v1', 
    'HLT_DiJetAve190_v1', 
    'HLT_DiJetAve240_v1', 
    'HLT_DiJetAve300_v1', 
    'HLT_DiJetAve30_v1', 
    'HLT_DiJetAve370_v1', 
    'HLT_DiJetAve60_v1', 
    'HLT_DiJetAve80_v1', 
    'HLT_Jet110_v2', 
    'HLT_Jet150_v2', 
    'HLT_Jet190_v2', 
    'HLT_Jet240_v2', 
    'HLT_Jet300_v1', 
    'HLT_Jet30_v2', 
    'HLT_Jet370_NoJetID_v2', 
    'HLT_Jet370_v2', 
    'HLT_Jet60_v2', 
    'HLT_Jet80_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMETBTag_selector
streamA_datasetMETBTag_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMETBTag_selector.l1tResults = cms.InputTag('')
streamA_datasetMETBTag_selector.throw      = cms.bool(False)
streamA_datasetMETBTag_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet110_Mu5_v3', 
    'HLT_BTagMu_DiJet20_Mu5_v3', 
    'HLT_BTagMu_DiJet40_Mu5_v3', 
    'HLT_BTagMu_DiJet70_Mu5_v3', 
    'HLT_CentralJet80_MET100_v2', 
    'HLT_CentralJet80_MET160_v2', 
    'HLT_CentralJet80_MET65_v2', 
    'HLT_CentralJet80_MET80_v2', 
    'HLT_DiJet60_MET45_v2', 
    'HLT_MET100_v2', 
    'HLT_MET120_v2', 
    'HLT_MET200_v2', 
    'HLT_PFMHT150_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v4', 
    'HLT_JetE30_NoBPTX_NoHalo_v4', 
    'HLT_JetE30_NoBPTX_v2', 
    'HLT_L1Tech_BSC_minBias_threshold1_v3', 
    'HLT_Physics_v1', 
    'HLT_PixelTracks_Multiplicity100_v2', 
    'HLT_PixelTracks_Multiplicity80_v2', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_Ele8_CaloIdL_TrkIdVL_v4', 
    'HLT_DoubleMu5_Ele8_v4', 
    'HLT_Mu10_Ele10_CaloIdL_v4', 
    'HLT_Mu15_DoublePhoton15_CaloIdL_v4', 
    'HLT_Mu15_Photon20_CaloIdL_v4', 
    'HLT_Mu17_Ele8_CaloIdL_v3', 
    'HLT_Mu5_DoubleEle8_v4', 
    'HLT_Mu5_Ele8_CaloIdL_TrkIdVL_Ele8_v4', 
    'HLT_Mu8_Ele17_CaloIdL_v3', 
    'HLT_Mu8_Photon20_CaloIdVT_IsoT_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_HT150_v1', 
    'HLT_DoubleMu3_HT200_v4', 
    'HLT_HT200_Mu5_PFMHT35_v2', 
    'HLT_HT250_Mu5_PFMHT35_v2', 
    'HLT_IsoMu17_CentralJet30_BTagIP_v2', 
    'HLT_Mu17_CentralJet30_BTagIP_v2', 
    'HLT_Mu17_CentralJet30_v3', 
    'HLT_Mu17_DiCentralJet30_v3', 
    'HLT_Mu17_TriCentralJet30_v3', 
    'HLT_Mu3_Ele8_CaloIdL_TrkIdVL_HT150_v1', 
    'HLT_Mu3_Ele8_CaloIdT_TrkIdVL_HT150_v1', 
    'HLT_Mu5_HT200_v5', 
    'HLT_Mu8_HT200_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Barrel_Upsilon_v1', 
    'HLT_Dimuon6p5_Barrel_Jpsi_v1', 
    'HLT_Dimuon6p5_Barrel_PsiPrime_v1', 
    'HLT_Dimuon6p5_Jpsi_Displaced_v1', 
    'HLT_Dimuon6p5_Jpsi_v1', 
    'HLT_Dimuon6p5_LowMass_Displaced_v1', 
    'HLT_Dimuon6p5_LowMass_v1', 
    'HLT_DoubleMu2_Bs_v2', 
    'HLT_Mu5_L2Mu2_Jpsi_v3', 
    'HLT_Mu5_Track2_Jpsi_v2', 
    'HLT_Mu7_Track7_Jpsi_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_DoubleJet30_ForwardBackward_v3', 
    'HLT_DoubleJet60_ForwardBackward_v3', 
    'HLT_DoubleJet70_ForwardBackward_v3', 
    'HLT_DoubleJet80_ForwardBackward_v3', 
    'HLT_ExclDiJet60_HFAND_v2', 
    'HLT_ExclDiJet60_HFOR_v2', 
    'HLT_L1DoubleJet36Central_v1', 
    'HLT_L1ETM30_v1', 
    'HLT_L1MultiJet_v1', 
    'HLT_QuadJet40_IsoPFTau40_v3', 
    'HLT_QuadJet40_v3', 
    'HLT_QuadJet50_BTagIP_v2', 
    'HLT_QuadJet50_Jet40_v2', 
    'HLT_QuadJet60_v2', 
    'HLT_QuadJet70_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoublePhoton33_v3', 
    'HLT_DoublePhoton5_IsoVL_CEP_v2', 
    'HLT_Photon125_NoSpikeFilter_v3', 
    'HLT_Photon20_CaloIdVL_IsoL_v2', 
    'HLT_Photon20_R9Id_Photon18_R9Id_v3', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_CaloIdL_IsoVL_v3', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_R9Id_v2', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_v3', 
    'HLT_Photon26_IsoVL_Photon18_IsoVL_v3', 
    'HLT_Photon26_IsoVL_Photon18_v3', 
    'HLT_Photon26_Photon18_v3', 
    'HLT_Photon26_R9Id_Photon18_CaloIdL_IsoVL_v2', 
    'HLT_Photon30_CaloIdVL_IsoL_v3', 
    'HLT_Photon30_CaloIdVL_v3', 
    'HLT_Photon32_CaloIdL_Photon26_CaloIdL_v3', 
    'HLT_Photon36_CaloIdL_Photon22_CaloIdL_v2', 
    'HLT_Photon50_CaloIdVL_IsoL_v2', 
    'HLT_Photon75_CaloIdVL_IsoL_v3', 
    'HLT_Photon75_CaloIdVL_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon60_CaloIdL_HT200_v3', 
    'HLT_Photon70_CaloIdL_HT200_v3', 
    'HLT_Photon70_CaloIdL_HT300_v3', 
    'HLT_Photon70_CaloIdL_MHT30_v3', 
    'HLT_Photon70_CaloIdL_MHT50_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v3', 
    'HLT_Ele32_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v2', 
    'HLT_Ele45_CaloIdVT_TrkIdT_v3', 
    'HLT_Ele90_NoSpikeFilter_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_IsoMu12_v2', 
    'HLT_IsoMu15_v6', 
    'HLT_IsoMu17_v6', 
    'HLT_IsoMu24_v2', 
    'HLT_IsoMu30_v2', 
    'HLT_L1SingleMu10_v1', 
    'HLT_L1SingleMu20_v1', 
    'HLT_L2Mu10_v2', 
    'HLT_L2Mu20_v2', 
    'HLT_Mu12_v2', 
    'HLT_Mu15_v3', 
    'HLT_Mu20_v2', 
    'HLT_Mu24_v2', 
    'HLT_Mu30_v2', 
    'HLT_Mu3_v4', 
    'HLT_Mu5_v4', 
    'HLT_Mu8_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleIsoPFTau20_Trk5_v4', 
    'HLT_IsoPFTau35_Trk20_MET45_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v4', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v4', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v3', 
    'HLT_Ele15_CaloIdVT_TrkIdT_LooseIsoPFTau15_v4', 
    'HLT_HT200_DoubleLooseIsoPFTau10_Trk3_PFMHT35_v2', 
    'HLT_HT250_DoubleLooseIsoPFTau10_Trk3_PFMHT35_v2', 
    'HLT_IsoMu12_LooseIsoPFTau10_v4', 
    'HLT_Mu15_LooseIsoPFTau20_v4')

