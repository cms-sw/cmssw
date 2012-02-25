# /dev/CMSSW_5_1_0/GRun/V97

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTag_selector
streamA_datasetBTag_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTag_selector.l1tResults = cms.InputTag('')
streamA_datasetBTag_selector.throw      = cms.bool(False)
streamA_datasetBTag_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet110_L1FastJet_Mu5_v1', 
    'HLT_BTagMu_DiJet20_L1FastJet_Mu5_v1', 
    'HLT_BTagMu_DiJet40_L1FastJet_Mu5_v1', 
    'HLT_BTagMu_DiJet70_L1FastJet_Mu5_v1', 
    'HLT_BTagMu_Jet300_L1FastJet_Mu5_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v9', 
    'HLT_BeamGas_HF_Beam1_v3', 
    'HLT_BeamGas_HF_Beam2_v3', 
    'HLT_IsoTrackHB_v10', 
    'HLT_IsoTrackHE_v11', 
    'HLT_L1SingleMuOpen_v5', 
    'HLT_L1Tech_DT_GlobalOR_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v9', 
    'HLT_L1SingleMuOpen_AntiBPTX_v4', 
    'HLT_L1TrackerCosmics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v1', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v11', 
    'HLT_Ele23_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT30_v1', 
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v1', 
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT15_v1', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_IsoMu5_v13', 
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v1', 
    'HLT_L2DoubleMu23_NoVertex_v9', 
    'HLT_Mu13_Mu8_v12', 
    'HLT_Mu17_Mu8_v12', 
    'HLT_Mu17_TkMu8_v5', 
    'HLT_Mu22_TkMu22_v1', 
    'HLT_Mu22_TkMu8_v1', 
    'HLT_TripleMu5_v14')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_BTagIPIter_v1', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_QuadCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_TrkIdT_QuadCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFJet30_v4', 
    'HLT_Ele27_WP80_CentralPFJet30_CentralPFJet25_PFMHT20_v1', 
    'HLT_Ele27_WP80_CentralPFJet30_CentralPFJet25_v1', 
    'HLT_Ele27_WP80_CentralPFJet80_v1', 
    'HLT_Ele27_WP80_PFJet30_PFJet25_Deta3_v1', 
    'HLT_Ele30_CaloIdVT_TrkIdT_PFJet100_PFJet25_v1', 
    'HLT_Ele30_CaloIdVT_TrkIdT_PFJet150_PFJet25_v1', 
    'HLT_MET80_Track50_dEdx3p6_v1', 
    'HLT_MET80_Track60_dEdx3p7_v1', 
    'HLT_MET80_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFEDMonitor_selector
streamA_datasetFEDMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFEDMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetFEDMonitor_selector.throw      = cms.bool(False)
streamA_datasetFEDMonitor_selector.triggerConditions = cms.vstring('HLT_DTErrors_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetForwardTriggers_selector
streamA_datasetForwardTriggers_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetForwardTriggers_selector.l1tResults = cms.InputTag('')
streamA_datasetForwardTriggers_selector.throw      = cms.bool(False)
streamA_datasetForwardTriggers_selector.triggerConditions = cms.vstring('HLT_L1Tech_CASTOR_HaloMuon_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHT_selector
streamA_datasetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHT_selector.throw      = cms.bool(False)
streamA_datasetHT_selector.triggerConditions = cms.vstring('HLT_HT250_L1FastJet_DoubleDisplacedPFJet60_ChgFraction10_v1', 
    'HLT_HT250_L1FastJet_DoubleDisplacedPFJet60_v1', 
    'HLT_HT250_L1FastJet_SingleDisplacedPFJet60_ChgFraction10_v1', 
    'HLT_HT250_L1FastJet_SingleDisplacedPFJet60_v1', 
    'HLT_HT750_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v6', 
    'HLT_L1Tech_HBHEHO_totalOR_v4', 
    'HLT_L1Tech_HCAL_HF_single_channel_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJet_selector
streamA_datasetJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJet_selector.l1tResults = cms.InputTag('')
streamA_datasetJet_selector.throw      = cms.bool(False)
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiPFJetAve140_v1', 
    'HLT_DiPFJetAve200_v1', 
    'HLT_DiPFJetAve260_v1', 
    'HLT_DiPFJetAve320_v1', 
    'HLT_DiPFJetAve400_v1', 
    'HLT_DiPFJetAve40_v1', 
    'HLT_DiPFJetAve80_v1', 
    'HLT_PFJet140_v1', 
    'HLT_PFJet200_v1', 
    'HLT_PFJet260_v1', 
    'HLT_PFJet320_v1', 
    'HLT_PFJet400_v1', 
    'HLT_PFJet40_v1', 
    'HLT_PFJet80_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetLogMonitor_selector
streamA_datasetLogMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetLogMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetLogMonitor_selector.throw      = cms.bool(False)
streamA_datasetLogMonitor_selector.triggerConditions = cms.vstring('HLT_LogMonitor_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMET_selector
streamA_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetMET_selector.throw      = cms.bool(False)
streamA_datasetMET_selector.triggerConditions = cms.vstring('HLT_DiPFJet40L1FastJet_PFMHTWOM65_M600VBF_LEADINGJETS_v1', 
    'HLT_DiPFJet40L1FastJet_PFMHTWOM65_M800VBF_ALLJETS_v1', 
    'HLT_IsoMu15_eta2p1_L1ETM20_v1', 
    'HLT_MonoCentralPFJet80L1FastJet_PFMHTWOM95_NHEF95_v1', 
    'HLT_Mu15_eta2p1_L1ETM20_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v11', 
    'HLT_JetE30_NoBPTX_v9', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v6', 
    'HLT_JetE70_NoBPTX3BX_NoHalo_v1', 
    'HLT_Physics_v3', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_Mu30_Ele30_CaloIdL_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleDisplacedMu4_DiPFJet40Neutral_L1FastJet_v1', 
    'HLT_Iso10Mu20_eta2p1_CentralPFJet30_BTagIPIter_v1', 
    'HLT_Iso10Mu20_eta2p1_CentralPFJet30_v1', 
    'HLT_Iso10Mu20_eta2p1_DiCentralPFJet30_v1', 
    'HLT_Iso10Mu20_eta2p1_QuadCentralPFJet30_v1', 
    'HLT_Iso10Mu20_eta2p1_TriCentralPFJet30_v1', 
    'HLT_IsoMu20_eta2p1_CentralPFJet80_v1', 
    'HLT_IsoMu24_eta2p1_CentralPFJet30_CentralPFJet25_PFMHT20_v1', 
    'HLT_IsoMu24_eta2p1_CentralPFJet30_CentralPFJet25_v1', 
    'HLT_IsoMu24_eta2p1_PFJet30_PFJet25_Deta3_v1', 
    'HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_L1FastJet_v1', 
    'HLT_Mu20_eta2p1_QuadCentralPFJet30_v1', 
    'HLT_Mu20_eta2p1_TriCentralPFJet30_v1', 
    'HLT_Mu24_eta2p1_CentralPFJet30_CentralPFJet25_v1', 
    'HLT_Mu24_eta2p1_PFJet30_PFJet25_Deta3_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_NoVertexing_v8', 
    'HLT_Dimuon0_Jpsi_v11', 
    'HLT_Dimuon0_PsiPrime_v1', 
    'HLT_Dimuon0_Upsilon_v11', 
    'HLT_Dimuon3p5_SameSign_v1', 
    'HLT_Dimuon5_Jpsi_v1', 
    'HLT_Dimuon5_PsiPrime_v1', 
    'HLT_Dimuon5_Upsilon_v1', 
    'HLT_Dimuon8_Upsilon_v1', 
    'HLT_Dimuon9_Jpsi_v1', 
    'HLT_Dimuon9_PsiPrime_v6', 
    'HLT_DoubleMu3p5_LowMass_Displaced_v1', 
    'HLT_DoubleMu4_Dimuon4_Bs_Barrel_v6', 
    'HLT_DoubleMu4_Dimuon6_Bs_v6', 
    'HLT_DoubleMu4_JpsiTk_Displaced_v1', 
    'HLT_DoubleMu4_Jpsi_Displaced_v6', 
    'HLT_Mu5_L2Mu3_Jpsi_v1', 
    'HLT_Mu5_Track2_Jpsi_v14', 
    'HLT_Mu5_Track3p5_Jpsi_v1', 
    'HLT_Tau2Mu_RegPixTrack_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_DiJet40Eta2p6_L1FastJet_BTagIP3D_v1', 
    'HLT_DiJet80Eta2p6_L1FastJet_BTagIP3DLoose_v1', 
    'HLT_EightJet35_L1FastJet_v3', 
    'HLT_EightJet40_L1FastJet_v3', 
    'HLT_ExclDiJet80_HFAND_v1', 
    'HLT_Jet160Eta2p4_Jet120Eta2p4_L1FastJet_DiBTagIP3DLoose_v1', 
    'HLT_Jet60Eta1p7_Jet53Eta1p7_L1FastJet_DiBTagIP3D_v1', 
    'HLT_Jet80Eta1p7_Jet70Eta1p7_L1FastJet_DiBTagIP3D_v1', 
    'HLT_L1DoubleJet36Central_v5', 
    'HLT_QuadJet70_L1FastJet_v1', 
    'HLT_QuadJet80_L1FastJet_v3', 
    'HLT_QuadJet90_L1FastJet_v1', 
    'HLT_QuadL1FastJet_BTagIP_VBF_v1', 
    'HLT_SixJet35_L1FastJet_v1', 
    'HLT_SixJet45_L1FastJet_v3', 
    'HLT_SixJet50_L1FastJet_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoubleEle33_CaloIdL_v8', 
    'HLT_DoubleEle33_CaloIdT_v4', 
    'HLT_DoublePhoton43_HEVT_v2', 
    'HLT_DoublePhoton48_HEVT_v2', 
    'HLT_DoublePhoton70_v2', 
    'HLT_DoublePhoton80_v3', 
    'HLT_Photon135_v3', 
    'HLT_Photon150_v1', 
    'HLT_Photon160_v1', 
    'HLT_Photon20_R9Id_Photon18_R9Id_v8', 
    'HLT_Photon250_NoHE_v1', 
    'HLT_Photon26_CaloId10_Iso50_Photon18_CaloId10_Iso50_Mass60_v1', 
    'HLT_Photon26_CaloId10_Iso50_Photon18_R9Id85_Mass60_v1', 
    'HLT_Photon26_Photon18_v8', 
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass60_v1', 
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_v1', 
    'HLT_Photon26_R9Id85_Photon18_CaloId10_Iso50_Mass60_v1', 
    'HLT_Photon26_R9Id85_Photon18_R9Id85_Mass60_v1', 
    'HLT_Photon300_NoHE_v1', 
    'HLT_Photon30_CaloIdVL_v9', 
    'HLT_Photon36_CaloId10_Iso50_Photon22_CaloId10_Iso50_v1', 
    'HLT_Photon36_CaloId10_Iso50_Photon22_R9Id85_v1', 
    'HLT_Photon36_Photon22_v2', 
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v1', 
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_v1', 
    'HLT_Photon36_R9Id85_Photon22_CaloId10_Iso50_v1', 
    'HLT_Photon36_R9Id85_Photon22_R9Id85_v1', 
    'HLT_Photon50_CaloIdVL_v5', 
    'HLT_Photon75_CaloIdVL_v8', 
    'HLT_Photon90_CaloIdVL_IsoL_v8', 
    'HLT_Photon90_CaloIdVL_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet25_v6', 
    'HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet30_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele100_CaloIdVT_TrkIdT_v4', 
    'HLT_Ele22_CaloIdL_CaloIsoVL_v1', 
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v4', 
    'HLT_Ele27_WP80_PFMT50_v10', 
    'HLT_Ele27_WP80_v4', 
    'HLT_Ele30_CaloIdVT_TrkIdT_v1', 
    'HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v4', 
    'HLT_Ele32_WP70_PFMT50_v10', 
    'HLT_Ele32_WP70_v4', 
    'HLT_Ele65_CaloIdVT_TrkIdT_v7', 
    'HLT_Ele80_CaloIdVT_TrkIdT_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_IsoMu20_eta2p1_v1', 
    'HLT_IsoMu24_eta2p1_v8', 
    'HLT_IsoMu30_eta2p1_v8', 
    'HLT_IsoMu34_eta2p1_v6', 
    'HLT_IsoMu40_eta2p1_v3', 
    'HLT_L1SingleMu12_v1', 
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v1', 
    'HLT_L2Mu20_NoVertex_NoBPTX3BX_NoHalo_v1', 
    'HLT_L2Mu20_eta2p1_NoVertex_v1', 
    'HLT_L2Mu30_NoVertex_NoBPTX3BX_NoHalo_v1', 
    'HLT_Mu12_v13', 
    'HLT_Mu15_eta2p1_v1', 
    'HLT_Mu17_v1', 
    'HLT_Mu24_eta2p1_v1', 
    'HLT_Mu30_eta2p1_v1', 
    'HLT_Mu40_eta2p1_Track50_dEdx3p6_v1', 
    'HLT_Mu40_eta2p1_Track60_dEdx3p7_v1', 
    'HLT_Mu40_eta2p1_v6', 
    'HLT_Mu50_eta2p1_v3', 
    'HLT_Mu5_v15', 
    'HLT_Mu8_v13')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20L1Jet_v1', 
    'HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v1', 
    'HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_v1', 
    'HLT_Ele20_CaloIdVT_TrkIdT_LooseIsoPFTau20_v1', 
    'HLT_IsoMu18_eta2p1_LooseIsoPFTau20_v1', 
    'HLT_Mu18_eta2p1_LooseIsoPFTau20_v1')

