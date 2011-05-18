# /dev/CMSSW_4_2_0/GRun/V75

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTag_selector
streamA_datasetBTag_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTag_selector.l1tResults = cms.InputTag('')
streamA_datasetBTag_selector.throw      = cms.bool(False)
streamA_datasetBTag_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet110_Mu5_v4', 
    'HLT_BTagMu_DiJet20_Mu5_v4', 
    'HLT_BTagMu_DiJet40_Mu5_v4', 
    'HLT_BTagMu_DiJet70_Mu5_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v3', 
    'HLT_BeamGas_BSC_v3', 
    'HLT_BeamGas_HF_v4', 
    'HLT_IsoTrackHB_v3', 
    'HLT_IsoTrackHE_v4', 
    'HLT_L1SingleEG12_v2', 
    'HLT_L1SingleEG5_v2', 
    'HLT_L1SingleJet16_v2', 
    'HLT_L1SingleJet36_v2', 
    'HLT_L1SingleMuOpen_DT_v2', 
    'HLT_L1SingleMuOpen_v2', 
    'HLT_L1_Interbunch_BSC_v2', 
    'HLT_L1_PreCollisions_v2', 
    'HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v3', 
    'HLT_L1SingleMuOpen_AntiBPTX_v2', 
    'HLT_L1TrackerCosmics_v3', 
    'HLT_L3MuonsCosmicTracking_v3', 
    'HLT_RegionalCosmicTracking_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_v4', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele15_HFL_v5', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele8_CaloIdL_CaloIsoVL_v4', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v4', 
    'HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v4', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass30_v2', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v4', 
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_v1', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v4', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v4', 
    'HLT_Ele8_CaloIdL_TrkIdVL_v4', 
    'HLT_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v3', 
    'HLT_Ele8_v4', 
    'HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v4', 
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_v5', 
    'HLT_DoubleMu4_Acoplanarity03_v3', 
    'HLT_DoubleMu5_Acoplanarity03_v1', 
    'HLT_DoubleMu6_v3', 
    'HLT_DoubleMu7_v3', 
    'HLT_L1DoubleMu0_v2', 
    'HLT_L2DoubleMu0_v4', 
    'HLT_L2DoubleMu23_NoVertex_v3', 
    'HLT_Mu13_Mu8_v2', 
    'HLT_Mu8_Jet40_v5', 
    'HLT_TripleMu5_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_DoubleEle8_CaloIdL_TrkIdVL_HT150_v2', 
    'HLT_DoubleEle8_CaloIdL_TrkIdVL_v1', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_HT150_v2', 
    'HLT_Ele10_CaloIdL_TrkIdVL_CaloIsoVL_TrkIsoVL_R005_MR200_v1', 
    'HLT_Ele10_CaloIdL_TrkIdVL_CaloIsoVL_TrkIsoVL_R020_MR200_v1', 
    'HLT_Ele10_CaloIdL_TrkIdVL_CaloIsoVL_TrkIsoVL_R025_MR200_v1', 
    'HLT_Ele10_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_R020_MR200_v1', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT200_v4', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_v4', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta2_v2', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta3_v2', 
    'HLT_Ele15_CaloIdVT_TrkIdT_Jet35_Jet25_Deta2_v2', 
    'HLT_Ele17_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_CentralJet25_PFMHT15_v2', 
    'HLT_Ele17_CaloIdVT_TrkIdT_CentralJet30_CentralJet25_v1', 
    'HLT_Ele25_CaloIdL_CaloIsoT_TrkIdVL_TrkIsoT_CentralJet30_BTagIP_v1', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_CentralJet25_PFMHT20_v2', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_BTagIP_v4', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_v4', 
    'HLT_Ele25_CaloIdVT_TrkIdT_DiCentralJet30_v3', 
    'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralJet30_v3', 
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v1', 
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v1', 
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v1', 
    'HLT_HT250_Ele5_CaloIdVL_TrkIdVL_CaloIsoVL_TrkIsoVL_PFMHT35_v4', 
    'HLT_HT300_Ele5_CaloIdVL_TrkIdVL_CaloIsoVL_TrkIsoVL_PFMHT40_v2', 
    'HLT_HT350_Ele5_CaloIdVL_TrkIdVL_CaloIsoVL_TrkIsoVL_PFMHT45_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHT_selector
streamA_datasetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHT_selector.throw      = cms.bool(False)
streamA_datasetHT_selector.triggerConditions = cms.vstring('HLT_DiJet130_PT130_v2', 
    'HLT_DiJet160_PT160_v2', 
    'HLT_HT150_AlphaT0p60_v3', 
    'HLT_HT150_v4', 
    'HLT_HT200_AlphaT0p53_v2', 
    'HLT_HT200_AlphaT0p60_v3', 
    'HLT_HT200_v4', 
    'HLT_HT250_AlphaT0p53_v2', 
    'HLT_HT250_AlphaT0p54_v2', 
    'HLT_HT250_DoubleDisplacedJet60_v3', 
    'HLT_HT250_MHT60_v4', 
    'HLT_HT250_MHT70_v1', 
    'HLT_HT250_MHT80_v1', 
    'HLT_HT250_v4', 
    'HLT_HT300_AlphaT0p52_v3', 
    'HLT_HT300_AlphaT0p53_v2', 
    'HLT_HT300_CentralJet30_BTagIP_PFMHT55_v2', 
    'HLT_HT300_CentralJet30_BTagIP_PFMHT75_v2', 
    'HLT_HT300_CentralJet30_BTagIP_v2', 
    'HLT_HT300_MHT75_v5', 
    'HLT_HT300_PFMHT55_v2', 
    'HLT_HT300_v5', 
    'HLT_HT350_AlphaT0p51_v3', 
    'HLT_HT350_AlphaT0p53_v3', 
    'HLT_HT350_v4', 
    'HLT_HT400_AlphaT0p51_v3', 
    'HLT_HT400_v4', 
    'HLT_HT450_v4', 
    'HLT_HT500_v4', 
    'HLT_HT550_v4', 
    'HLT_R014_MR150_CentralJet40_BTagIP_v2', 
    'HLT_R014_MR150_v1', 
    'HLT_R014_MR450_CentralJet40_BTagIP_v2', 
    'HLT_R020_MR150_v1', 
    'HLT_R020_MR350_CentralJet40_BTagIP_v2', 
    'HLT_R020_MR500_v1', 
    'HLT_R020_MR550_v1', 
    'HLT_R025_MR150_v1', 
    'HLT_R025_MR250_CentralJet40_BTagIP_v2', 
    'HLT_R025_MR400_v1', 
    'HLT_R025_MR450_v1', 
    'HLT_R033_MR300_v1', 
    'HLT_R033_MR350_v1', 
    'HLT_R038_MR200_v1', 
    'HLT_R038_MR250_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v3', 
    'HLT_L1Tech_HBHEHO_totalOR_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v4', 
    'HLT_HcalPhiSym_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJet_selector
streamA_datasetJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJet_selector.l1tResults = cms.InputTag('')
streamA_datasetJet_selector.throw      = cms.bool(False)
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiJetAve110_v3', 
    'HLT_DiJetAve150_v3', 
    'HLT_DiJetAve190_v3', 
    'HLT_DiJetAve240_v3', 
    'HLT_DiJetAve300_v3', 
    'HLT_DiJetAve30_v3', 
    'HLT_DiJetAve370_v3', 
    'HLT_DiJetAve60_v3', 
    'HLT_DiJetAve80_v3', 
    'HLT_Jet110_v3', 
    'HLT_Jet150_v3', 
    'HLT_Jet190_v3', 
    'HLT_Jet240_v3', 
    'HLT_Jet300_v2', 
    'HLT_Jet30_v3', 
    'HLT_Jet370_NoJetID_v3', 
    'HLT_Jet370_v3', 
    'HLT_Jet60_v3', 
    'HLT_Jet80_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMET_selector
streamA_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetMET_selector.throw      = cms.bool(False)
streamA_datasetMET_selector.triggerConditions = cms.vstring('HLT_CentralJet80_MET100_v3', 
    'HLT_CentralJet80_MET160_v3', 
    'HLT_CentralJet80_MET65_v3', 
    'HLT_CentralJet80_MET80HF_v2', 
    'HLT_DiCentralJet20_BTagIP_MET65_v2', 
    'HLT_DiCentralJet20_MET80_v1', 
    'HLT_DiJet60_MET45_v3', 
    'HLT_L2Mu60_1Hit_MET40_v1', 
    'HLT_L2Mu60_1Hit_MET60_v1', 
    'HLT_MET100_HBHENoiseFiltered_v1', 
    'HLT_MET100_v3', 
    'HLT_MET120_HBHENoiseFiltered_v1', 
    'HLT_MET120_v3', 
    'HLT_MET200_HBHENoiseFiltered_v1', 
    'HLT_MET200_v3', 
    'HLT_PFMHT150_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v5', 
    'HLT_JetE30_NoBPTX_NoHalo_v5', 
    'HLT_JetE30_NoBPTX_v3', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v1', 
    'HLT_L1Tech_BSC_minBias_threshold1_v4', 
    'HLT_Physics_v1', 
    'HLT_PixelTracks_Multiplicity100_v3', 
    'HLT_PixelTracks_Multiplicity80_v3', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_Ele8_CaloIdL_TrkIdVL_v5', 
    'HLT_DoubleMu5_Ele8_v5', 
    'HLT_Mu15_DoublePhoton15_CaloIdL_v5', 
    'HLT_Mu15_Photon20_CaloIdL_v5', 
    'HLT_Mu17_Ele8_CaloIdL_v4', 
    'HLT_Mu5_DoubleEle8_CaloIdL_TrkIdVL_v1', 
    'HLT_Mu8_Ele17_CaloIdL_v4', 
    'HLT_Mu8_Photon20_CaloIdVT_IsoT_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_HT150_v2', 
    'HLT_DoubleMu3_HT200_v5', 
    'HLT_HT250_Mu15_PFMHT20_v2', 
    'HLT_HT250_Mu5_PFMHT35_v4', 
    'HLT_HT300_Mu5_PFMHT40_v2', 
    'HLT_HT350_Mu5_PFMHT45_v2', 
    'HLT_IsoMu17_CentralJet30_BTagIP_v4', 
    'HLT_Mu12_DiCentralJet30_BTagIP3D_v1', 
    'HLT_Mu15_HT200_v2', 
    'HLT_Mu17_CentralJet30_BTagIP_v4', 
    'HLT_Mu17_CentralJet30_v5', 
    'HLT_Mu17_DiCentralJet30_v5', 
    'HLT_Mu17_TriCentralJet30_v5', 
    'HLT_Mu20_HT200_v2', 
    'HLT_Mu3_DiJet30_v1', 
    'HLT_Mu3_Ele8_CaloIdL_TrkIdVL_HT150_v2', 
    'HLT_Mu3_Ele8_CaloIdT_TrkIdVL_HT150_v2', 
    'HLT_Mu3_QuadJet30_v1', 
    'HLT_Mu3_TriJet30_v1', 
    'HLT_Mu8_R005_MR200_v1', 
    'HLT_Mu8_R020_MR200_v1', 
    'HLT_Mu8_R025_MR200_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_Muon_v1', 
    'HLT_Dimuon0_Jpsi_v1', 
    'HLT_Dimuon0_Upsilon_Muon_v1', 
    'HLT_Dimuon0_Upsilon_v1', 
    'HLT_Dimuon10_Jpsi_Barrel_v1', 
    'HLT_Dimuon4_Bs_Barrel_v2', 
    'HLT_Dimuon5_Upsilon_Barrel_v1', 
    'HLT_Dimuon6_Bs_v1', 
    'HLT_Dimuon7_Jpsi_Displaced_v1', 
    'HLT_Dimuon7_Jpsi_X_Barrel_v1', 
    'HLT_Dimuon7_LowMass_Displaced_v1', 
    'HLT_Dimuon7_PsiPrime_v1', 
    'HLT_DoubleMu2_Bs_v3', 
    'HLT_Mu5_L2Mu2_Jpsi_v4', 
    'HLT_Mu5_Track2_Jpsi_v4', 
    'HLT_Mu7_Track7_Jpsi_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_CentralJet46_BTagIP3D_CentralJet38_BTagIP3D_v1', 
    'HLT_DoubleJet30_ForwardBackward_v4', 
    'HLT_DoubleJet60_ForwardBackward_v4', 
    'HLT_DoubleJet70_ForwardBackward_v4', 
    'HLT_DoubleJet80_ForwardBackward_v4', 
    'HLT_ExclDiJet60_HFAND_v3', 
    'HLT_ExclDiJet60_HFOR_v3', 
    'HLT_L1DoubleJet36Central_v2', 
    'HLT_L1ETM30_v2', 
    'HLT_L1MultiJet_v2', 
    'HLT_QuadJet40_IsoPFTau40_v5', 
    'HLT_QuadJet40_v4', 
    'HLT_QuadJet50_BTagIP_v4', 
    'HLT_QuadJet50_Jet40_v3', 
    'HLT_QuadJet60_v3', 
    'HLT_QuadJet70_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoubleEle33_CaloIdL_v1', 
    'HLT_DoubleEle33_v1', 
    'HLT_DoublePhoton33_HEVT_v1', 
    'HLT_DoublePhoton33_v4', 
    'HLT_DoublePhoton40_MR150_v1', 
    'HLT_DoublePhoton40_R014_MR150_v1', 
    'HLT_DoublePhoton50_v1', 
    'HLT_DoublePhoton5_IsoVL_CEP_v3', 
    'HLT_DoublePhoton60_v1', 
    'HLT_Photon125_v1', 
    'HLT_Photon200_NoHE_v1', 
    'HLT_Photon20_CaloIdVL_IsoL_v3', 
    'HLT_Photon20_R9Id_Photon18_R9Id_v4', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_CaloIdL_IsoVL_v4', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_R9Id_v3', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_v4', 
    'HLT_Photon26_IsoVL_Photon18_IsoVL_v4', 
    'HLT_Photon26_IsoVL_Photon18_v4', 
    'HLT_Photon26_Photon18_v4', 
    'HLT_Photon26_R9Id_Photon18_CaloIdL_IsoVL_v3', 
    'HLT_Photon26_R9Id_Photon18_R9Id_v1', 
    'HLT_Photon30_CaloIdVL_IsoL_v4', 
    'HLT_Photon30_CaloIdVL_v4', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_v1', 
    'HLT_Photon36_CaloIdL_Photon22_CaloIdL_v3', 
    'HLT_Photon36_IsoVL_Photon22_v1', 
    'HLT_Photon40_CaloIdL_Photon28_CaloIdL_v1', 
    'HLT_Photon50_CaloIdVL_IsoL_v3', 
    'HLT_Photon50_CaloIdVL_v1', 
    'HLT_Photon75_CaloIdVL_IsoL_v4', 
    'HLT_Photon75_CaloIdVL_v4', 
    'HLT_Photon90_CaloIdVL_IsoL_v1', 
    'HLT_Photon90_CaloIdVL_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon40_R005_MR150_v1', 
    'HLT_Photon40_R014_MR450_v1', 
    'HLT_Photon40_R020_MR300_v1', 
    'HLT_Photon40_R025_MR200_v1', 
    'HLT_Photon40_R038_MR150_v1', 
    'HLT_Photon70_CaloIdL_HT300_v4', 
    'HLT_Photon70_CaloIdL_HT350_v3', 
    'HLT_Photon70_CaloIdL_MHT50_v4', 
    'HLT_Photon70_CaloIdL_MHT70_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele32_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_v1', 
    'HLT_Ele32_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v3', 
    'HLT_Ele52_CaloIdVT_TrkIdT_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_IsoMu12_v4', 
    'HLT_IsoMu15_v8', 
    'HLT_IsoMu17_v8', 
    'HLT_IsoMu24_v4', 
    'HLT_IsoMu30_v4', 
    'HLT_L1SingleMu10_v2', 
    'HLT_L1SingleMu20_v2', 
    'HLT_L2Mu10_v3', 
    'HLT_L2Mu20_v3', 
    'HLT_Mu12_v3', 
    'HLT_Mu15_v4', 
    'HLT_Mu20_v3', 
    'HLT_Mu24_v3', 
    'HLT_Mu30_v3', 
    'HLT_Mu3_v5', 
    'HLT_Mu40_v1', 
    'HLT_Mu5_v5', 
    'HLT_Mu8_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleIsoPFTau35_Trk5_eta2p1_v1', 
    'HLT_DoubleIsoPFTau40_Trk5_eta2p1_v1', 
    'HLT_IsoPFTau35_Trk20_MET60_v1', 
    'HLT_IsoPFTau35_Trk20_v1', 
    'HLT_IsoPFTau45_Trk20_MET60_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v4', 
    'HLT_Ele15_CaloIdVT_TrkIdT_LooseIsoPFTau20_v1', 
    'HLT_Ele18_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v1', 
    'HLT_HT250_DoubleIsoPFTau10_Trk3_PFMHT35_v1', 
    'HLT_HT300_DoubleIsoPFTau10_Trk3_PFMHT40_v1', 
    'HLT_HT350_DoubleIsoPFTau10_Trk3_PFMHT45_v1', 
    'HLT_IsoMu15_LooseIsoPFTau15_v2', 
    'HLT_IsoMu15_LooseIsoPFTau20_v1', 
    'HLT_IsoMu15_TightIsoPFTau20_v1', 
    'HLT_Mu15_LooseIsoPFTau15_v2')

