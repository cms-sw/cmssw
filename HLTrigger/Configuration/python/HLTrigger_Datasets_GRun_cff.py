# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream PhysicsEGammaCommissioning Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v2', 
    'HLT_IsoTrackHB_v1', 
    'HLT_IsoTrackHE_v1', 
    'HLT_L1SingleMuOpen_DT_v2', 
    'HLT_L1Tech_DT_GlobalOR_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleEG_selector
streamA_datasetDoubleEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleEG_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleEG_selector.throw      = cms.bool(False)
streamA_datasetDoubleEG_selector.triggerConditions = cms.vstring('HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_DoublePixelVeto_Mass55_v1', 
    'HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_DoublePixelVeto_Mass55_v1', 
    'HLT_Diphoton30_18_R9Id_OR_IsoCaloId_AND_HE_R9Id_DoublePixelSeedMatch_Mass70_v1', 
    'HLT_Diphoton30_18_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v1', 
    'HLT_Diphoton30_18_Solid_R9Id_AND_IsoCaloId_AND_HE_R9Id_Mass55_v1', 
    'HLT_DoubleEle24_22_eta2p1_WPLoose_Gsf_v1', 
    'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v2', 
    'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v2', 
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300_v2', 
    'HLT_DoublePhoton85_v2', 
    'HLT_ECALHT800_v2', 
    'HLT_Ele10_CaloIdM_TrackIdM_CentralPFJet30_BTagCSV0p54PF_v2', 
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v2', 
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_v2', 
    'HLT_Ele12_CaloIdM_TrackIdM_PFJet30_v2', 
    'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v2', 
    'HLT_Ele17_CaloIdL_TrackIdL_IsoVL_v1', 
    'HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v2', 
    'HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_v2', 
    'HLT_Ele18_CaloIdL_TrackIdL_IsoVL_PFJet30_v2', 
    'HLT_Ele18_CaloIdM_TrackIdM_PFJet30_v2', 
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v2', 
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v2', 
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v2', 
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v2', 
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v2', 
    'HLT_Ele30WP60_Ele8_Mass55_v1', 
    'HLT_Ele30WP60_SC4_Mass55_v2', 
    'HLT_Ele33_CaloIdL_TrackIdL_IsoVL_PFJet30_v2', 
    'HLT_Ele33_CaloIdM_TrackIdM_PFJet30_v2', 
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v2', 
    'HLT_Photon26_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon16_AND_HE10_R9Id65_Eta2_Mass60_v2', 
    'HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon22_AND_HE10_R9Id65_Eta2_Mass15_v2', 
    'HLT_Photon42_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon25_AND_HE10_R9Id65_Eta2_Mass15_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFullTrack_selector
streamA_datasetFullTrack_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFullTrack_selector.l1tResults = cms.InputTag('')
streamA_datasetFullTrack_selector.throw      = cms.bool(False)
streamA_datasetFullTrack_selector.triggerConditions = cms.vstring('HLT_FullTrack12ForEndOfFill_v1', 
    'HLT_FullTrack50_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINCaloJets_selector
streamA_datasetHINCaloJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINCaloJets_selector.l1tResults = cms.InputTag('')
streamA_datasetHINCaloJets_selector.throw      = cms.bool(False)
streamA_datasetHINCaloJets_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet100_v2', 
    'HLT_AK4CaloJet30_v2', 
    'HLT_AK4CaloJet40_v2', 
    'HLT_AK4CaloJet50_v2', 
    'HLT_AK4CaloJet80_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINPFJets_selector
streamA_datasetHINPFJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINPFJets_selector.l1tResults = cms.InputTag('')
streamA_datasetHINPFJets_selector.throw      = cms.bool(False)
streamA_datasetHINPFJets_selector.triggerConditions = cms.vstring('HLT_AK4PFJet100_v2', 
    'HLT_AK4PFJet30_v2', 
    'HLT_AK4PFJet50_v2', 
    'HLT_AK4PFJet80_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINPhoton_selector
streamA_datasetHINPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetHINPhoton_selector.throw      = cms.bool(False)
streamA_datasetHINPhoton_selector.triggerConditions = cms.vstring('HLT_HISinglePhoton10ForEndOfFill_v1', 
    'HLT_HISinglePhoton15ForEndOfFill_v1', 
    'HLT_HISinglePhoton20ForEndOfFill_v1', 
    'HLT_HISinglePhoton40_v2', 
    'HLT_HISinglePhoton60_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHLTPhysics_selector
streamA_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamA_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamA_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v2', 
    'HLT_HcalPhiSym_v2', 
    'HLT_HcalUTCA_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighMultiplicity_selector
streamA_datasetHighMultiplicity_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighMultiplicity_selector.l1tResults = cms.InputTag('')
streamA_datasetHighMultiplicity_selector.throw      = cms.bool(False)
streamA_datasetHighMultiplicity_selector.triggerConditions = cms.vstring('HLT_PixelTracks_Multiplicity110ForEndOfFill_v1', 
    'HLT_PixelTracks_Multiplicity135ForEndOfFill_v1', 
    'HLT_PixelTracks_Multiplicity160ForEndOfFill_v1', 
    'HLT_PixelTracks_Multiplicity60ForEndOfFill_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighMultiplicity85_selector
streamA_datasetHighMultiplicity85_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighMultiplicity85_selector.l1tResults = cms.InputTag('')
streamA_datasetHighMultiplicity85_selector.throw      = cms.bool(False)
streamA_datasetHighMultiplicity85_selector.triggerConditions = cms.vstring('HLT_PixelTracks_Multiplicity85ForEndOfFill_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMonteCarlo_selector
streamA_datasetMonteCarlo_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMonteCarlo_selector.l1tResults = cms.InputTag('')
streamA_datasetMonteCarlo_selector.throw      = cms.bool(False)
streamA_datasetMonteCarlo_selector.triggerConditions = cms.vstring('MC_AK4CaloJets_v1', 
    'MC_AK4PFJets_v1', 
    'MC_AK8CaloHT_v1', 
    'MC_AK8PFHT_v1', 
    'MC_AK8PFJets_v1', 
    'MC_AK8TrimPFJets_v1', 
    'MC_CaloHT_v1', 
    'MC_CaloMET_JetIdCleaned_v1', 
    'MC_CaloMET_v1', 
    'MC_CaloMHT_v1', 
    'MC_Diphoton10_10_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass10_v1', 
    'MC_DoubleEle5_CaloIdL_GsfTrkIdVL_MW_v1', 
    'MC_DoubleGlbTrkMu_TrkIsoVVL_DZ_v1', 
    'MC_DoubleMuNoFiltersNoVtx_v1', 
    'MC_DoubleMu_TrkIsoVVL_DZ_v1', 
    'MC_Ele15_Ele10_CaloIdL_TrackIdL_IsoVL_DZ_v1', 
    'MC_Ele5_WPLoose_Gsf_v1', 
    'MC_IsoMu_v1', 
    'MC_IsoTkMu15_v1', 
    'MC_PFHT_v1', 
    'MC_PFMET_v1', 
    'MC_PFMHT_v1', 
    'MC_ReducedIterativeTracking_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetNoBPTX_selector
streamA_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamA_datasetNoBPTX_selector.throw      = cms.bool(False)
streamA_datasetNoBPTX_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v2', 
    'HLT_JetE30_NoBPTX_v2', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v2', 
    'HLT_JetE70_NoBPTX3BX_NoHalo_v2', 
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v2', 
    'HLT_L2Mu10_NoVertex_NoBPTX_v2', 
    'HLT_L2Mu35_NoVertex_3Sta_NoBPTX3BX_NoHalo_v2', 
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_NoHalo_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele105_CaloIdVT_GsfTrkIdT_v2', 
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v1', 
    'HLT_Ele15_IsoVVVL_BTagCSV0p72_PFHT400_v2', 
    'HLT_Ele15_IsoVVVL_PFHT350_PFMET50_v1', 
    'HLT_Ele15_IsoVVVL_PFHT350_v1', 
    'HLT_Ele15_IsoVVVL_PFHT600_v2', 
    'HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v1', 
    'HLT_Ele22_eta2p1_WPLoose_Gsf_v1', 
    'HLT_Ele22_eta2p1_WPTight_Gsf_v1', 
    'HLT_Ele23_WPLoose_Gsf_v1', 
    'HLT_Ele27_WPLoose_Gsf_WHbbBoost_v1', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v1', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v1', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_HT200_v1', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v1', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_TriCentralPFJet30_v1', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_TriCentralPFJet50_40_30_v1', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_v1', 
    'HLT_Ele27_eta2p1_WPTight_Gsf_v1', 
    'HLT_Ele32_eta2p1_WPLoose_Gsf_CentralPFJet30_BTagCSV07_v1', 
    'HLT_Ele32_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v1', 
    'HLT_Ele32_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v1', 
    'HLT_Ele32_eta2p1_WPLoose_Gsf_TriCentralPFJet30_v1', 
    'HLT_Ele32_eta2p1_WPLoose_Gsf_TriCentralPFJet50_40_30_v1', 
    'HLT_Ele32_eta2p1_WPLoose_Gsf_v1', 
    'HLT_Ele32_eta2p1_WPTight_Gsf_v1', 
    'HLT_Ele45_CaloIdVT_GsfTrkIdT_PFJet200_PFJet50_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSinglePhoton_selector
streamA_datasetSinglePhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSinglePhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetSinglePhoton_selector.throw      = cms.bool(False)
streamA_datasetSinglePhoton_selector.triggerConditions = cms.vstring('HLT_Photon120_R9Id90_HE10_Iso40_EBOnly_PFMET40_v2', 
    'HLT_Photon120_R9Id90_HE10_Iso40_EBOnly_VBF_v2', 
    'HLT_Photon120_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon120_v2', 
    'HLT_Photon135_PFMET100_JetIdCleaned_v1', 
    'HLT_Photon165_HE10_v2', 
    'HLT_Photon165_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon175_v2', 
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_PFMET40_v2', 
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_VBF_v2', 
    'HLT_Photon22_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon22_v2', 
    'HLT_Photon250_NoHE_v2', 
    'HLT_Photon300_NoHE_v2', 
    'HLT_Photon30_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon30_v2', 
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v2', 
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_VBF_v2', 
    'HLT_Photon36_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon36_v2', 
    'HLT_Photon500_v1', 
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_PFMET40_v2', 
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_VBF_v2', 
    'HLT_Photon50_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon50_v2', 
    'HLT_Photon600_v1', 
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_PFMET40_v2', 
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_VBF_v2', 
    'HLT_Photon75_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon75_v2', 
    'HLT_Photon90_CaloIdL_PFHT500_v2', 
    'HLT_Photon90_CaloIdL_PFHT600_v1', 
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_PFMET40_v2', 
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_VBF_v2', 
    'HLT_Photon90_R9Id90_HE10_IsoM_v2', 
    'HLT_Photon90_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetZeroBias_selector
streamA_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamA_datasetZeroBias_selector.throw      = cms.bool(False)
streamA_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_Random_v1', 
    'HLT_ZeroBias_v2')


# dump of the Stream PhysicsHadronsTaus Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTagCSV_selector
streamA_datasetBTagCSV_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTagCSV_selector.l1tResults = cms.InputTag('')
streamA_datasetBTagCSV_selector.throw      = cms.bool(False)
streamA_datasetBTagCSV_selector.triggerConditions = cms.vstring('HLT_DoubleJet90_Double30_DoubleBTagCSV0p67_v2', 
    'HLT_DoubleJet90_Double30_TripleBTagCSV0p67_v2', 
    'HLT_DoubleJetsC100_DoubleBTagCSV0p85_DoublePFJetsC160_v1', 
    'HLT_DoubleJetsC100_DoubleBTagCSV0p9_DoublePFJetsC100MaxDeta1p6_v1', 
    'HLT_DoubleJetsC112_DoubleBTagCSV0p85_DoublePFJetsC172_v1', 
    'HLT_DoubleJetsC112_DoubleBTagCSV0p9_DoublePFJetsC112MaxDeta1p6_v1', 
    'HLT_QuadJet45_DoubleBTagCSV0p67_v2', 
    'HLT_QuadJet45_TripleBTagCSV0p67_v2', 
    'HLT_QuadPFJet_DoubleBTagCSV_VBF_Mqq200_v2', 
    'HLT_QuadPFJet_DoubleBTagCSV_VBF_Mqq240_v2', 
    'HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq460_v2', 
    'HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTagMu_selector
streamA_datasetBTagMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTagMu_selector.l1tResults = cms.InputTag('')
streamA_datasetBTagMu_selector.throw      = cms.bool(False)
streamA_datasetBTagMu_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet110_Mu5_v2', 
    'HLT_BTagMu_DiJet20_Mu5_v2', 
    'HLT_BTagMu_DiJet40_Mu5_v2', 
    'HLT_BTagMu_DiJet70_Mu5_v2', 
    'HLT_BTagMu_Jet300_Mu5_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDisplacedJet_selector
streamA_datasetDisplacedJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDisplacedJet_selector.l1tResults = cms.InputTag('')
streamA_datasetDisplacedJet_selector.throw      = cms.bool(False)
streamA_datasetDisplacedJet_selector.triggerConditions = cms.vstring('HLT_HT250_DisplacedDijet40_DisplacedTrack_v2', 
    'HLT_HT350_DisplacedDijet40_DisplacedTrack_v2', 
    'HLT_HT350_DisplacedDijet80_DisplacedTrack_v2', 
    'HLT_HT350_DisplacedDijet80_Tight_DisplacedTrack_v2', 
    'HLT_HT400_DisplacedDijet40_Inclusive_v2', 
    'HLT_HT500_DisplacedDijet40_Inclusive_v2', 
    'HLT_HT550_DisplacedDijet40_Inclusive_v2', 
    'HLT_HT650_DisplacedDijet80_Inclusive_v2', 
    'HLT_HT750_DisplacedDijet80_Inclusive_v2', 
    'HLT_VBF_DisplacedJet40_DisplacedTrack_2TrackIP2DSig5_v2', 
    'HLT_VBF_DisplacedJet40_DisplacedTrack_v2', 
    'HLT_VBF_DisplacedJet40_Hadronic_2PromptTrack_v2', 
    'HLT_VBF_DisplacedJet40_Hadronic_v2', 
    'HLT_VBF_DisplacedJet40_TightID_DisplacedTrack_v2', 
    'HLT_VBF_DisplacedJet40_TightID_Hadronic_v2', 
    'HLT_VBF_DisplacedJet40_VTightID_DisplacedTrack_v2', 
    'HLT_VBF_DisplacedJet40_VTightID_Hadronic_v2', 
    'HLT_VBF_DisplacedJet40_VVTightID_DisplacedTrack_v2', 
    'HLT_VBF_DisplacedJet40_VVTightID_Hadronic_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHTMHT_selector
streamA_datasetHTMHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHTMHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHTMHT_selector.throw      = cms.bool(False)
streamA_datasetHTMHT_selector.triggerConditions = cms.vstring('HLT_DiCentralPFJet55_PFMET110_JetIdCleaned_v1', 
    'HLT_DiPFJet40_DEta3p5_MJJ600_PFMETNoMu140_JetIdCleaned_v1', 
    'HLT_DiPFJet40_DEta3p5_MJJ600_PFMETNoMu80_JetIdCleaned_v1', 
    'HLT_PFHT200_DiPFJetAve90_PFAlphaT0p57_v1', 
    'HLT_PFHT200_DiPFJetAve90_PFAlphaT0p63_v1', 
    'HLT_PFHT250_DiPFJetAve90_PFAlphaT0p55_v1', 
    'HLT_PFHT250_DiPFJetAve90_PFAlphaT0p58_v1', 
    'HLT_PFHT300_DiPFJetAve90_PFAlphaT0p53_v1', 
    'HLT_PFHT300_DiPFJetAve90_PFAlphaT0p54_v1', 
    'HLT_PFHT350_DiPFJetAve90_PFAlphaT0p52_v1', 
    'HLT_PFHT350_DiPFJetAve90_PFAlphaT0p53_v1', 
    'HLT_PFHT350_PFMET100_JetIdCleaned_v1', 
    'HLT_PFHT400_DiPFJetAve90_PFAlphaT0p51_v1', 
    'HLT_PFHT400_DiPFJetAve90_PFAlphaT0p52_v1', 
    'HLT_Rsq0p25_v1', 
    'HLT_Rsq0p30_v1', 
    'HLT_RsqMR240_Rsq0p09_MR200_4jet_v1', 
    'HLT_RsqMR240_Rsq0p09_MR200_v1', 
    'HLT_RsqMR270_Rsq0p09_MR200_4jet_v1', 
    'HLT_RsqMR270_Rsq0p09_MR200_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetHT_selector
streamA_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetJetHT_selector.throw      = cms.bool(False)
streamA_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV0p45_v1', 
    'HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV0p45_v2', 
    'HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV0p45_v1', 
    'HLT_AK8PFHT650_TrimR0p1PT0p03Mass50_v1', 
    'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v2', 
    'HLT_AK8PFJet360_TrimMass30_v2', 
    'HLT_CaloJet500_NoJetID_v2', 
    'HLT_DiPFJetAve100_HFJEC_v2', 
    'HLT_DiPFJetAve140_v1', 
    'HLT_DiPFJetAve160_HFJEC_v2', 
    'HLT_DiPFJetAve200_v1', 
    'HLT_DiPFJetAve220_HFJEC_v2', 
    'HLT_DiPFJetAve260_v1', 
    'HLT_DiPFJetAve300_HFJEC_v2', 
    'HLT_DiPFJetAve320_v1', 
    'HLT_DiPFJetAve400_v1', 
    'HLT_DiPFJetAve40_v1', 
    'HLT_DiPFJetAve500_v1', 
    'HLT_DiPFJetAve60_HFJEC_v2', 
    'HLT_DiPFJetAve60_v1', 
    'HLT_DiPFJetAve80_HFJEC_v2', 
    'HLT_DiPFJetAve80_v1', 
    'HLT_HT2000_v1', 
    'HLT_HT200_v1', 
    'HLT_HT2500_v1', 
    'HLT_HT275_v1', 
    'HLT_HT325_v1', 
    'HLT_HT425_v1', 
    'HLT_HT575_v1', 
    'HLT_L1_TripleJet_VBF_v2', 
    'HLT_PFHT200_v1', 
    'HLT_PFHT250_v1', 
    'HLT_PFHT300_v1', 
    'HLT_PFHT350_v2', 
    'HLT_PFHT400_SixJet30_BTagCSV0p55_2PFBTagCSV0p72_v2', 
    'HLT_PFHT400_SixJet30_v2', 
    'HLT_PFHT400_v1', 
    'HLT_PFHT450_SixJet40_PFBTagCSV0p72_v2', 
    'HLT_PFHT450_SixJet40_v2', 
    'HLT_PFHT475_v1', 
    'HLT_PFHT550_4Jet_v2', 
    'HLT_PFHT600_v2', 
    'HLT_PFHT650_4Jet_v2', 
    'HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v2', 
    'HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v2', 
    'HLT_PFHT650_v2', 
    'HLT_PFHT750_4JetPt50_v1', 
    'HLT_PFHT800_v1', 
    'HLT_PFJet140_v2', 
    'HLT_PFJet200_v2', 
    'HLT_PFJet260_v2', 
    'HLT_PFJet320_v2', 
    'HLT_PFJet400_v2', 
    'HLT_PFJet40_v2', 
    'HLT_PFJet450_v2', 
    'HLT_PFJet500_v2', 
    'HLT_PFJet60_v2', 
    'HLT_PFJet80_v2', 
    'HLT_QuadPFJet_VBF_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMET_selector
streamA_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetMET_selector.throw      = cms.bool(False)
streamA_datasetMET_selector.triggerConditions = cms.vstring('HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDTight_BTagCSV0p72_v1', 
    'HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDTight_v1', 
    'HLT_MET200_JetIdCleaned_v1', 
    'HLT_MET250_v1', 
    'HLT_MET300_v1', 
    'HLT_MET60_IsoTrk35_Loose_v1', 
    'HLT_MET75_IsoTrk50_v2', 
    'HLT_MET90_IsoTrk50_v2', 
    'HLT_MonoCentralPFJet80_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v1', 
    'HLT_MonoCentralPFJet80_PFMETNoMu90_JetIdCleaned_PFMHTNoMu90_IDTight_v1', 
    'HLT_Mu14er_PFMET100_JetIdCleaned_v1', 
    'HLT_Mu3er_PFHT140_PFMET125_JetIdCleaned_v1', 
    'HLT_Mu6_PFHT200_PFMET100_JetIdCleaned_v1', 
    'HLT_Mu6_PFHT200_PFMET80_JetIdCleaned_BTagCSV0p72_v1', 
    'HLT_PFMET100_PFMHT100_IDTight_v1', 
    'HLT_PFMET110_PFMHT110_IDTight_v1', 
    'HLT_PFMET120_JetIdCleaned_BTagCSV0p72_v1', 
    'HLT_PFMET120_JetIdCleaned_Mu5_v1', 
    'HLT_PFMET120_PFMHT120_IDTight_v1', 
    'HLT_PFMET170_HBHECleaned_v1', 
    'HLT_PFMET170_JetIdCleaned_v1', 
    'HLT_PFMET170_NoiseCleaned_v2', 
    'HLT_PFMET170_v1', 
    'HLT_PFMET300_JetIdCleaned_v1', 
    'HLT_PFMET400_JetIdCleaned_v1', 
    'HLT_PFMET90_PFMHT90_IDTight_v1', 
    'HLT_PFMETNoMu120_JetIdCleaned_PFMHTNoMu120_IDTight_v1', 
    'HLT_PFMETNoMu90_JetIdCleaned_PFMHTNoMu90_IDTight_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v2', 
    'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120_JetIdCleaned_v1', 
    'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_JetIdCleaned_v1', 
    'HLT_LooseIsoPFTau50_Trk30_eta2p1_v2')


# dump of the Stream PhysicsMuons Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCharmonium_selector
streamA_datasetCharmonium_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCharmonium_selector.l1tResults = cms.InputTag('')
streamA_datasetCharmonium_selector.throw      = cms.bool(False)
streamA_datasetCharmonium_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_Muon_v2', 
    'HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing_v2', 
    'HLT_Dimuon0er16_Jpsi_NoVertexing_v2', 
    'HLT_Dimuon10_Jpsi_Barrel_v2', 
    'HLT_Dimuon13_PsiPrime_v2', 
    'HLT_Dimuon16_Jpsi_v2', 
    'HLT_Dimuon20_Jpsi_v2', 
    'HLT_Dimuon6_Jpsi_NoVertexing_v2', 
    'HLT_Dimuon8_PsiPrime_Barrel_v2', 
    'HLT_DoubleMu4_3_Bs_v2', 
    'HLT_DoubleMu4_3_Jpsi_Displaced_v2', 
    'HLT_DoubleMu4_JpsiTrk_Displaced_v2', 
    'HLT_DoubleMu4_PsiPrimeTrk_Displaced_v2', 
    'HLT_Mu7p5_L2Mu2_Jpsi_v2', 
    'HLT_Mu7p5_Track2_Jpsi_v2', 
    'HLT_Mu7p5_Track3p5_Jpsi_v2', 
    'HLT_Mu7p5_Track7_Jpsi_v2', 
    'HLT_QuadMuon0_Dimuon0_Jpsi_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMuon_selector
streamA_datasetDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMuon_selector.throw      = cms.bool(False)
streamA_datasetDoubleMuon_selector.triggerConditions = cms.vstring('HLT_DoubleMu18NoFiltersNoVtx_v2', 
    'HLT_DoubleMu23NoFiltersNoVtxDisplaced_v2', 
    'HLT_DoubleMu28NoFiltersNoVtxDisplaced_v2', 
    'HLT_DoubleMu33NoFiltersNoVtx_v2', 
    'HLT_DoubleMu38NoFiltersNoVtx_v2', 
    'HLT_DoubleMu8_Mass8_PFHT300_v2', 
    'HLT_L2DoubleMu23_NoVertex_v2', 
    'HLT_L2DoubleMu28_NoVertex_2Cha_Angle2p5_Mass10_v2', 
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_Mass10_v2', 
    'HLT_Mu10_CentralPFJet30_BTagCSV0p54PF_v2', 
    'HLT_Mu17_Mu8_DZ_v2', 
    'HLT_Mu17_Mu8_SameSign_DZ_v1', 
    'HLT_Mu17_Mu8_v1', 
    'HLT_Mu17_TkMu8_DZ_v2', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v2', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v2', 
    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v2', 
    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v2', 
    'HLT_Mu17_TrkIsoVVL_v2', 
    'HLT_Mu17_v2', 
    'HLT_Mu20_Mu10_DZ_v1', 
    'HLT_Mu20_Mu10_SameSign_DZ_v1', 
    'HLT_Mu20_Mu10_v1', 
    'HLT_Mu24_TrkIsoVVL_v2', 
    'HLT_Mu24_v2', 
    'HLT_Mu27_TkMu8_v2', 
    'HLT_Mu30_TkMu11_v2', 
    'HLT_Mu34_TrkIsoVVL_v2', 
    'HLT_Mu34_v2', 
    'HLT_Mu40_TkMu11_v2', 
    'HLT_Mu8_TrkIsoVVL_v2', 
    'HLT_Mu8_v2', 
    'HLT_TripleMu_12_10_5_v2', 
    'HLT_TrkMu15_DoubleTrkMu5NoFiltersNoVtx_v2', 
    'HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMuonLowMass_selector
streamA_datasetDoubleMuonLowMass_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMuonLowMass_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMuonLowMass_selector.throw      = cms.bool(False)
streamA_datasetDoubleMuonLowMass_selector.triggerConditions = cms.vstring('HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Phi_Barrel_v2', 
    'HLT_Dimuon0_Upsilon_Muon_v2', 
    'HLT_Dimuon13_Upsilon_v2', 
    'HLT_Dimuon8_Upsilon_Barrel_v2', 
    'HLT_Mu16_TkMu0_dEta18_Onia_v2', 
    'HLT_Mu16_TkMu0_dEta18_Phi_v2', 
    'HLT_Mu25_TkMu0_dEta18_Onia_v2', 
    'HLT_Mu7p5_L2Mu2_Upsilon_v2', 
    'HLT_Mu7p5_Track2_Upsilon_v2', 
    'HLT_Mu7p5_Track3p5_Upsilon_v2', 
    'HLT_Mu7p5_Track7_Upsilon_v2', 
    'HLT_QuadMuon0_Dimuon0_Upsilon_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuonEG_selector
streamA_datasetMuonEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuonEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuonEG_selector.throw      = cms.bool(False)
streamA_datasetMuonEG_selector.triggerConditions = cms.vstring('HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v2', 
    'HLT_Mu12_Photon25_CaloIdL_L1ISO_v2', 
    'HLT_Mu12_Photon25_CaloIdL_L1OR_v2', 
    'HLT_Mu12_Photon25_CaloIdL_v2', 
    'HLT_Mu17_Photon30_CaloIdL_L1ISO_v2', 
    'HLT_Mu17_Photon35_CaloIdL_L1ISO_v2', 
    'HLT_Mu17_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v2', 
    'HLT_Mu23NoFiltersNoVtx_Photon23_CaloIdL_v2', 
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v2', 
    'HLT_Mu28NoFiltersNoVtxDisplaced_Photon28_CaloIdL_v2', 
    'HLT_Mu30_Ele30_CaloIdL_GsfTrkIdVL_v2', 
    'HLT_Mu33NoFiltersNoVtxDisplaced_Photon33_CaloIdL_v2', 
    'HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v2', 
    'HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v2', 
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v2', 
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT300_v2', 
    'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v2', 
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMuon_selector
streamA_datasetSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMuon_selector.throw      = cms.bool(False)
streamA_datasetSingleMuon_selector.triggerConditions = cms.vstring('HLT_DoubleIsoMu17_eta2p1_v2', 
    'HLT_IsoMu16_eta2p1_MET30_JetIdCleaned_LooseIsoPFTau50_Trk30_eta2p1_v1', 
    'HLT_IsoMu16_eta2p1_MET30_JetIdCleaned_v1', 
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_SingleL1_v2', 
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v2', 
    'HLT_IsoMu17_eta2p1_MediumIsoPFTau40_Trk1_eta2p1_Reg_v2', 
    'HLT_IsoMu17_eta2p1_v2', 
    'HLT_IsoMu20_eta2p1_CentralPFJet30_BTagCSV07_v2', 
    'HLT_IsoMu20_eta2p1_TriCentralPFJet30_v2', 
    'HLT_IsoMu20_eta2p1_TriCentralPFJet50_40_30_v2', 
    'HLT_IsoMu20_eta2p1_v2', 
    'HLT_IsoMu20_v2', 
    'HLT_IsoMu24_eta2p1_CentralPFJet30_BTagCSV07_v2', 
    'HLT_IsoMu24_eta2p1_LooseIsoPFTau20_v2', 
    'HLT_IsoMu24_eta2p1_TriCentralPFJet30_v2', 
    'HLT_IsoMu24_eta2p1_TriCentralPFJet50_40_30_v2', 
    'HLT_IsoMu24_eta2p1_v2', 
    'HLT_IsoMu27_v2', 
    'HLT_IsoTkMu20_eta2p1_v2', 
    'HLT_IsoTkMu20_v2', 
    'HLT_IsoTkMu24_eta2p1_v2', 
    'HLT_IsoTkMu27_v2', 
    'HLT_L1SingleMu16_v1', 
    'HLT_L1SingleMuOpen_v2', 
    'HLT_L2Mu10_v1', 
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_JetIdCleaned_v1', 
    'HLT_Mu15_IsoVVVL_BTagCSV0p72_PFHT400_v2', 
    'HLT_Mu15_IsoVVVL_PFHT350_PFMET50_v1', 
    'HLT_Mu15_IsoVVVL_PFHT350_v1', 
    'HLT_Mu15_IsoVVVL_PFHT600_v2', 
    'HLT_Mu16_eta2p1_MET30_JetIdCleaned_v1', 
    'HLT_Mu20_v2', 
    'HLT_Mu24_eta2p1_v2', 
    'HLT_Mu27_v2', 
    'HLT_Mu28NoFiltersNoVtx_CentralCaloJet40_v2', 
    'HLT_Mu28NoFiltersNoVtx_DisplacedJet40_Loose_v2', 
    'HLT_Mu300_v1', 
    'HLT_Mu33NoFiltersNoVtxDisplaced_DisplacedJet50_Loose_v2', 
    'HLT_Mu33NoFiltersNoVtxDisplaced_DisplacedJet50_Tight_v2', 
    'HLT_Mu350_v1', 
    'HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Loose_v2', 
    'HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Tight_v2', 
    'HLT_Mu38NoFiltersNoVtx_DisplacedJet60_Loose_v2', 
    'HLT_Mu40_eta2p1_PFJet200_PFJet50_v2', 
    'HLT_Mu45_eta2p1_v2', 
    'HLT_Mu50_v2', 
    'HLT_Mu55_v1', 
    'HLT_TkMu20_v2', 
    'HLT_TkMu24_eta2p1_v2', 
    'HLT_TkMu27_v2')

