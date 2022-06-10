# /dev/CMSSW_12_4_0/GRun

import FWCore.ParameterSet.Config as cms


# stream ParkingBPH1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingBPH1_datasetParkingBPH1_selector
streamParkingBPH1_datasetParkingBPH1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingBPH1_datasetParkingBPH1_selector.l1tResults = cms.InputTag('')
streamParkingBPH1_datasetParkingBPH1_selector.throw      = cms.bool(False)
streamParkingBPH1_datasetParkingBPH1_selector.triggerConditions = cms.vstring(
    'HLT_Mu12_IP6_part0_v2',
    'HLT_Mu7_IP4_part0_v2',
    'HLT_Mu8_IP3_part0_v3',
    'HLT_Mu8_IP5_part0_v2',
    'HLT_Mu8_IP6_part0_v2',
    'HLT_Mu9_IP0_part0_v2',
    'HLT_Mu9_IP3_part0_v2',
    'HLT_Mu9_IP4_part0_v2',
    'HLT_Mu9_IP5_part0_v2',
    'HLT_Mu9_IP6_part0_v3'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingBPH1_datasetParkingBPHPromptCSCS_selector
streamParkingBPH1_datasetParkingBPHPromptCSCS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingBPH1_datasetParkingBPHPromptCSCS_selector.l1tResults = cms.InputTag('')
streamParkingBPH1_datasetParkingBPHPromptCSCS_selector.throw      = cms.bool(False)
streamParkingBPH1_datasetParkingBPHPromptCSCS_selector.triggerConditions = cms.vstring(
    'HLT_Mu12_IP6_ToCSCS_v1',
    'HLT_Mu7_IP4_ToCSCS_v1',
    'HLT_Mu8_IP3_ToCSCS_v1',
    'HLT_Mu8_IP5_ToCSCS_v1',
    'HLT_Mu8_IP6_ToCSCS_v1',
    'HLT_Mu9_IP4_ToCSCS_v1',
    'HLT_Mu9_IP5_ToCSCS_v1',
    'HLT_Mu9_IP6_ToCSCS_v1'
)


# stream ParkingBPH2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingBPH2_datasetParkingBPH2_selector
streamParkingBPH2_datasetParkingBPH2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingBPH2_datasetParkingBPH2_selector.l1tResults = cms.InputTag('')
streamParkingBPH2_datasetParkingBPH2_selector.throw      = cms.bool(False)
streamParkingBPH2_datasetParkingBPH2_selector.triggerConditions = cms.vstring(
    'HLT_Mu12_IP6_part1_v2',
    'HLT_Mu7_IP4_part1_v2',
    'HLT_Mu8_IP3_part1_v3',
    'HLT_Mu8_IP5_part1_v2',
    'HLT_Mu8_IP6_part1_v2',
    'HLT_Mu9_IP4_part1_v2',
    'HLT_Mu9_IP5_part1_v2',
    'HLT_Mu9_IP6_part1_v3'
)


# stream ParkingBPH3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingBPH3_datasetParkingBPH3_selector
streamParkingBPH3_datasetParkingBPH3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingBPH3_datasetParkingBPH3_selector.l1tResults = cms.InputTag('')
streamParkingBPH3_datasetParkingBPH3_selector.throw      = cms.bool(False)
streamParkingBPH3_datasetParkingBPH3_selector.triggerConditions = cms.vstring(
    'HLT_Mu12_IP6_part2_v2',
    'HLT_Mu7_IP4_part2_v2',
    'HLT_Mu8_IP3_part2_v3',
    'HLT_Mu8_IP5_part2_v2',
    'HLT_Mu8_IP6_part2_v2',
    'HLT_Mu9_IP4_part2_v2',
    'HLT_Mu9_IP5_part2_v2',
    'HLT_Mu9_IP6_part2_v3'
)


# stream ParkingBPH4

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingBPH4_datasetParkingBPH4_selector
streamParkingBPH4_datasetParkingBPH4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingBPH4_datasetParkingBPH4_selector.l1tResults = cms.InputTag('')
streamParkingBPH4_datasetParkingBPH4_selector.throw      = cms.bool(False)
streamParkingBPH4_datasetParkingBPH4_selector.triggerConditions = cms.vstring(
    'HLT_Mu12_IP6_part3_v2',
    'HLT_Mu7_IP4_part3_v2',
    'HLT_Mu8_IP3_part3_v3',
    'HLT_Mu8_IP5_part3_v2',
    'HLT_Mu8_IP6_part3_v2',
    'HLT_Mu9_IP4_part3_v2',
    'HLT_Mu9_IP5_part3_v2',
    'HLT_Mu9_IP6_part3_v3'
)


# stream ParkingBPH5

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingBPH5_datasetParkingBPH5_selector
streamParkingBPH5_datasetParkingBPH5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingBPH5_datasetParkingBPH5_selector.l1tResults = cms.InputTag('')
streamParkingBPH5_datasetParkingBPH5_selector.throw      = cms.bool(False)
streamParkingBPH5_datasetParkingBPH5_selector.triggerConditions = cms.vstring(
    'HLT_Mu12_IP6_part4_v2',
    'HLT_Mu7_IP4_part4_v2',
    'HLT_Mu8_IP3_part4_v3',
    'HLT_Mu8_IP5_part4_v2',
    'HLT_Mu8_IP6_part4_v2',
    'HLT_Mu9_IP4_part4_v2',
    'HLT_Mu9_IP5_part4_v2',
    'HLT_Mu9_IP6_part4_v3'
)


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring(
    'HLT_IsoTrackHB_v4',
    'HLT_IsoTrackHE_v4',
    'HLT_L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCosmics_selector
streamPhysicsCommissioning_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCosmics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_L1SingleMuCosmics_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring(
    'HLT_HcalNZS_v13',
    'HLT_HcalPhiSym_v15'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.triggerConditions = cms.vstring(
    'HLT_SinglePhoton10_Eta3p1ForPPRef_v8',
    'HLT_SinglePhoton20_Eta3p1ForPPRef_v9'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.triggerConditions = cms.vstring('HLT_SinglePhoton30_Eta3p1ForPPRef_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetIsolatedBunch_selector
streamPhysicsCommissioning_datasetIsolatedBunch_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetIsolatedBunch_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetIsolatedBunch_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetIsolatedBunch_selector.triggerConditions = cms.vstring('HLT_HcalIsolatedbunch_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMonteCarlo_selector
streamPhysicsCommissioning_datasetMonteCarlo_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMonteCarlo_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMonteCarlo_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMonteCarlo_selector.triggerConditions = cms.vstring(
    'MC_AK4CaloJetsFromPV_v8',
    'MC_AK4CaloJets_v9',
    'MC_AK4PFJets_v17',
    'MC_AK8CaloHT_v8',
    'MC_AK8PFHT_v16',
    'MC_AK8PFJets_v17',
    'MC_AK8TrimPFJets_v17',
    'MC_CaloBTagDeepCSV_v8',
    'MC_CaloHT_v8',
    'MC_CaloMET_JetIdCleaned_v9',
    'MC_CaloMET_v8',
    'MC_CaloMHT_v8',
    'MC_Diphoton10_10_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass10_v13',
    'MC_DoubleEle5_CaloIdL_MW_v15',
    'MC_DoubleMuNoFiltersNoVtx_v7',
    'MC_DoubleMu_TrkIsoVVL_DZ_v11',
    'MC_Ele15_Ele10_CaloIdL_TrackIdL_IsoVL_DZ_v15',
    'MC_Ele5_WPTight_Gsf_v8',
    'MC_IsoMu_v15',
    'MC_PFBTagDeepCSV_v10',
    'MC_PFBTagDeepJet_v1',
    'MC_PFHT_v16',
    'MC_PFMET_v17',
    'MC_PFMHT_v16',
    'MC_ReducedIterativeTracking_v12',
    'MC_Run3_PFScoutingPixelTracking_v16'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring(
    'HLT_CDC_L2cosmic_10_er1p0_v1',
    'HLT_CDC_L2cosmic_5p5_er1p0_v1',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v5',
    'HLT_L2Mu10_NoVertex_NoBPTX_v6',
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v5',
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v4',
    'HLT_UncorrectedJetE30_NoBPTX3BX_v6',
    'HLT_UncorrectedJetE30_NoBPTX_v6',
    'HLT_UncorrectedJetE60_NoBPTX3BX_v6',
    'HLT_UncorrectedJetE70_NoBPTX3BX_v6'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring(
    'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v1',
    'HLT_ZeroBias_FirstBXAfterTrain_v3',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v5',
    'HLT_ZeroBias_FirstCollisionInTrain_v4',
    'HLT_ZeroBias_IsolatedBunches_v5',
    'HLT_ZeroBias_LastCollisionInTrain_v3',
    'HLT_ZeroBias_v6'
)


# stream PhysicsEGamma

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetEGamma_selector
streamPhysicsEGamma_datasetEGamma_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetEGamma_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetEGamma_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetEGamma_selector.triggerConditions = cms.vstring(
    'HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v4',
    'HLT_DiPhoton10Time1ns_v1',
    'HLT_DiPhoton10Time1p2ns_v1',
    'HLT_DiPhoton10Time1p4ns_v1',
    'HLT_DiPhoton10Time1p6ns_v1',
    'HLT_DiPhoton10Time1p8ns_v1',
    'HLT_DiPhoton10Time2ns_v1',
    'HLT_DiPhoton10_CaloIdL_v1',
    'HLT_DiPhoton10sminlt0p12_v1',
    'HLT_DiPhoton10sminlt0p14_v1',
    'HLT_DiPhoton10sminlt0p16_v1',
    'HLT_DiPhoton10sminlt0p1_v1',
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v13',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v1',
    'HLT_Diphoton20_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v1',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v1',
    'HLT_Diphoton22_14_eta1p5_R9IdL_AND_HE_AND_IsoTCaloIdT_v1',
    'HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_Mass55_v13',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v2',
    'HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v2',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v13',
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v13',
    'HLT_DoubleEle25_CaloIdL_MW_v4',
    'HLT_DoubleEle27_CaloIdL_MW_v4',
    'HLT_DoubleEle33_CaloIdL_MW_v17',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v20',
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v20',
    'HLT_DoublePhoton33_CaloIdL_v6',
    'HLT_DoublePhoton70_v6',
    'HLT_DoublePhoton85_v14',
    'HLT_ECALHT800_v10',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v14',
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v18',
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v7',
    'HLT_Ele145_CaloIdVT_GsfTrkIdT_v8',
    'HLT_Ele15_CaloIdL_TrackIdL_IsoVL_PFJet30_v3',
    'HLT_Ele15_Ele8_CaloIdL_TrackIdL_IsoVL_v3',
    'HLT_Ele15_IsoVVVL_PFHT450_CaloBTagDeepCSV_4p5_v8',
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v16',
    'HLT_Ele15_IsoVVVL_PFHT450_v16',
    'HLT_Ele15_IsoVVVL_PFHT600_v20',
    'HLT_Ele15_WPLoose_Gsf_v3',
    'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v9',
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v16',
    'HLT_Ele17_WPLoose_Gsf_v3',
    'HLT_Ele200_CaloIdVT_GsfTrkIdT_v8',
    'HLT_Ele20_WPLoose_Gsf_v6',
    'HLT_Ele20_WPTight_Gsf_v6',
    'HLT_Ele20_eta2p1_WPLoose_Gsf_v6',
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v18',
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v18',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v19',
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v19',
    'HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1_v1',
    'HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTauHPS30_eta2p1_CrossL1_v1',
    'HLT_Ele250_CaloIdVT_GsfTrkIdT_v13',
    'HLT_Ele27_Ele37_CaloIdL_MW_v4',
    'HLT_Ele27_WPTight_Gsf_v16',
    'HLT_Ele28_HighEta_SC20_Mass55_v13',
    'HLT_Ele28_WPTight_Gsf_v1',
    'HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v13',
    'HLT_Ele300_CaloIdVT_GsfTrkIdT_v13',
    'HLT_Ele30_WPTight_Gsf_v1',
    'HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v13',
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v9',
    'HLT_Ele32_WPTight_Gsf_v15',
    'HLT_Ele35_WPTight_Gsf_L1EGMT_v5',
    'HLT_Ele35_WPTight_Gsf_v9',
    'HLT_Ele38_WPTight_Gsf_v9',
    'HLT_Ele40_WPTight_Gsf_v9',
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v18',
    'HLT_Ele50_IsoVVVL_PFHT450_v16',
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v16',
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v18',
    'HLT_Photon100EBHE10_v2',
    'HLT_Photon100EB_TightID_TightIso_v2',
    'HLT_Photon100EEHE10_v2',
    'HLT_Photon100EE_TightID_TightIso_v2',
    'HLT_Photon110EB_TightID_TightIso_v2',
    'HLT_Photon120EB_TightID_TightIso_v2',
    'HLT_Photon120_R9Id90_HE10_IsoM_v14',
    'HLT_Photon120_v13',
    'HLT_Photon150_v6',
    'HLT_Photon165_R9Id90_HE10_IsoM_v15',
    'HLT_Photon175_v14',
    'HLT_Photon200_v13',
    'HLT_Photon20_HoverELoose_v10',
    'HLT_Photon20_v2',
    'HLT_Photon300_NoHE_v12',
    'HLT_Photon30EB_TightID_TightIso_v2',
    'HLT_Photon30_HoverELoose_v10',
    'HLT_Photon33_v5',
    'HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50_v5',
    'HLT_Photon50_R9Id90_HE10_IsoM_v14',
    'HLT_Photon50_v13',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15_v11',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_v5',
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_v5',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ300_PFJetsMJJ400DEta3_v5',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_CaloMJJ400_PFJetsMJJ600DEta3_v5',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v5',
    'HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ600DEta3_v5',
    'HLT_Photon75_R9Id90_HE10_IsoM_v14',
    'HLT_Photon75_v13',
    'HLT_Photon90_CaloIdL_PFHT700_v16',
    'HLT_Photon90_R9Id90_HE10_IsoM_v14',
    'HLT_Photon90_v13'
)


# stream PhysicsEndOfFill

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetEmptyBX_selector
streamPhysicsEndOfFill_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetEmptyBX_selector.triggerConditions = cms.vstring(
    'HLT_L1NotBptxOR_v3',
    'HLT_L1UnpairedBunchBptxMinus_v2',
    'HLT_L1UnpairedBunchBptxPlus_v2'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetFSQJet1_selector
streamPhysicsEndOfFill_datasetFSQJet1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetFSQJet1_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetFSQJet1_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetFSQJet1_selector.triggerConditions = cms.vstring(
    'HLT_DiPFJet15_NoCaloMatched_v16',
    'HLT_DiPFJet25_NoCaloMatched_v16'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetFSQJet2_selector
streamPhysicsEndOfFill_datasetFSQJet2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetFSQJet2_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetFSQJet2_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetFSQJet2_selector.triggerConditions = cms.vstring(
    'HLT_DiPFJet15_FBEta3_NoCaloMatched_v17',
    'HLT_DiPFJet25_FBEta3_NoCaloMatched_v17',
    'HLT_DiPFJetAve15_HFJEC_v17',
    'HLT_DiPFJetAve25_HFJEC_v17',
    'HLT_DiPFJetAve35_HFJEC_v17'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHINCaloJets_selector
streamPhysicsEndOfFill_datasetHINCaloJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHINCaloJets_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHINCaloJets_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHINCaloJets_selector.triggerConditions = cms.vstring(
    'HLT_AK4CaloJet100_v10',
    'HLT_AK4CaloJet120_v9',
    'HLT_AK4CaloJet30_v11',
    'HLT_AK4CaloJet40_v10',
    'HLT_AK4CaloJet50_v10',
    'HLT_AK4CaloJet80_v10'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHINPFJets_selector
streamPhysicsEndOfFill_datasetHINPFJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHINPFJets_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHINPFJets_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHINPFJets_selector.triggerConditions = cms.vstring(
    'HLT_AK4PFJet100_v19',
    'HLT_AK4PFJet120_v18',
    'HLT_AK4PFJet30_v19',
    'HLT_AK4PFJet50_v19',
    'HLT_AK4PFJet80_v19'
)


# stream PhysicsHLTPhysics0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics0_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics0_datasetEphemeralHLTPhysics1_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')


# stream PhysicsHLTPhysics1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics2_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics1_datasetEphemeralHLTPhysics3_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')


# stream PhysicsHLTPhysics2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics4_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics2_datasetEphemeralHLTPhysics5_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')


# stream PhysicsHLTPhysics3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics6_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.l1tResults = cms.InputTag('')
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.throw      = cms.bool(False)
streamPhysicsHLTPhysics3_datasetEphemeralHLTPhysics7_selector.triggerConditions = cms.vstring('HLT_EphemeralPhysics_v1')


# stream PhysicsHadronsTaus

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetBTagMu_selector
streamPhysicsHadronsTaus_datasetBTagMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetBTagMu_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetBTagMu_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetBTagMu_selector.triggerConditions = cms.vstring(
    'HLT_BTagMu_AK4DiJet110_Mu5_v13',
    'HLT_BTagMu_AK4DiJet170_Mu5_v12',
    'HLT_BTagMu_AK4DiJet20_Mu5_v13',
    'HLT_BTagMu_AK4DiJet40_Mu5_v13',
    'HLT_BTagMu_AK4DiJet70_Mu5_v13',
    'HLT_BTagMu_AK4Jet300_Mu5_v12',
    'HLT_BTagMu_AK8DiJet170_Mu5_v9',
    'HLT_BTagMu_AK8Jet170_DoubleMu5_v2',
    'HLT_BTagMu_AK8Jet300_Mu5_v12'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetDisplacedJet_selector
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.triggerConditions = cms.vstring(
    'HLT_CaloMET60_DTCluster50_v1',
    'HLT_CaloMET60_DTClusterNoMB1S50_v1',
    'HLT_CscCluster_Loose_v1',
    'HLT_CscCluster_Medium_v1',
    'HLT_CscCluster_Tight_v1',
    'HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v1',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_v1',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay1nsInclusive_v1',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_v1',
    'HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay2nsInclusive_v1',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_v1',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet35_Inclusive1PtrkShortSig5_v1',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v1',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet40_Inclusive1PtrkShortSig5_v1',
    'HLT_HT200_L1SingleLLPJet_DisplacedDijet60_DisplacedTrack_v1',
    'HLT_HT270_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v1',
    'HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_v1',
    'HLT_HT400_DisplacedDijet40_DisplacedTrack_v13',
    'HLT_HT420_L1SingleLLPJet_DisplacedDijet60_Inclusive_v1',
    'HLT_HT425_v9',
    'HLT_HT430_DelayedJet40_DoubleDelay0p5nsTrackless_v1',
    'HLT_HT430_DelayedJet40_DoubleDelay1nsInclusive_v1',
    'HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v1',
    'HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v1',
    'HLT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_v1',
    'HLT_HT430_DisplacedDijet35_Inclusive1PtrkShortSig5_v1',
    'HLT_HT430_DisplacedDijet40_DisplacedTrack_v13',
    'HLT_HT430_DisplacedDijet40_Inclusive1PtrkShortSig5_v1',
    'HLT_HT430_DisplacedDijet60_DisplacedTrack_v13',
    'HLT_HT500_DisplacedDijet40_DisplacedTrack_v13',
    'HLT_HT550_DisplacedDijet60_Inclusive_v13',
    'HLT_HT650_DisplacedDijet60_Inclusive_v13',
    'HLT_L1CSCShower_DTCluster50_v1',
    'HLT_L1CSCShower_DTCluster75_v1',
    'HLT_L1MET_DTCluster50_v1',
    'HLT_L1MET_DTClusterNoMB1S50_v1',
    'HLT_L1Mu6HT240_v1',
    'HLT_Mu6HT240_DisplacedDijet30_Inclusive0PtrkShortSig5_v1',
    'HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v1',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive0PtrkShortSig5_v1',
    'HLT_Mu6HT240_DisplacedDijet35_Inclusive1PtrkShortSig5_DisplacedLoose_v1',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive0PtrkShortSig5_v1',
    'HLT_Mu6HT240_DisplacedDijet40_Inclusive1PtrkShortSig5_DisplacedLoose_v1'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetJetHT_selector
streamPhysicsHadronsTaus_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetJetHT_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetJetHT_selector.triggerConditions = cms.vstring(
    'HLT_AK8DiPFJet250_250_MassSD30_v1',
    'HLT_AK8DiPFJet250_250_MassSD50_v1',
    'HLT_AK8DiPFJet260_260_MassSD30_v1',
    'HLT_AK8DiPFJet270_270_MassSD30_v1',
    'HLT_AK8PFHT750_TrimMass50_v12',
    'HLT_AK8PFHT800_TrimMass50_v12',
    'HLT_AK8PFHT850_TrimMass50_v11',
    'HLT_AK8PFHT900_TrimMass50_v11',
    'HLT_AK8PFJet140_v15',
    'HLT_AK8PFJet15_v3',
    'HLT_AK8PFJet200_v15',
    'HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35_v1',
    'HLT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetTauTau0p30_v1',
    'HLT_AK8PFJet230_SoftDropMass40_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35_v1',
    'HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetTauTau0p30_v1',
    'HLT_AK8PFJet25_v3',
    'HLT_AK8PFJet260_v16',
    'HLT_AK8PFJet275_SoftDropMass40_PFAK8ParticleNetBB0p35_v1',
    'HLT_AK8PFJet275_SoftDropMass40_PFAK8ParticleNetTauTau0p30_v1',
    'HLT_AK8PFJet320_v16',
    'HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p17_v2',
    'HLT_AK8PFJet330_TrimMass30_PFAK8BTagDeepCSV_p1_v2',
    'HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np2_v2',
    'HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v2',
    'HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02_v3',
    'HLT_AK8PFJet360_TrimMass30_v18',
    'HLT_AK8PFJet380_TrimMass30_v11',
    'HLT_AK8PFJet400_MassSD30_v1',
    'HLT_AK8PFJet400_SoftDropMass40_v1',
    'HLT_AK8PFJet400_TrimMass30_v12',
    'HLT_AK8PFJet400_v16',
    'HLT_AK8PFJet40_v16',
    'HLT_AK8PFJet420_MassSD30_v1',
    'HLT_AK8PFJet420_TrimMass30_v11',
    'HLT_AK8PFJet425_SoftDropMass40_v1',
    'HLT_AK8PFJet450_MassSD30_v1',
    'HLT_AK8PFJet450_SoftDropMass40_v1',
    'HLT_AK8PFJet450_v16',
    'HLT_AK8PFJet500_v16',
    'HLT_AK8PFJet550_v11',
    'HLT_AK8PFJet60_v15',
    'HLT_AK8PFJet80_v15',
    'HLT_AK8PFJetFwd140_v14',
    'HLT_AK8PFJetFwd15_v3',
    'HLT_AK8PFJetFwd200_v14',
    'HLT_AK8PFJetFwd25_v3',
    'HLT_AK8PFJetFwd260_v15',
    'HLT_AK8PFJetFwd320_v15',
    'HLT_AK8PFJetFwd400_v15',
    'HLT_AK8PFJetFwd40_v15',
    'HLT_AK8PFJetFwd450_v15',
    'HLT_AK8PFJetFwd500_v15',
    'HLT_AK8PFJetFwd60_v14',
    'HLT_AK8PFJetFwd80_v14',
    'HLT_CaloJet500_NoJetID_v12',
    'HLT_CaloJet550_NoJetID_v7',
    'HLT_DiPFJetAve100_HFJEC_v16',
    'HLT_DiPFJetAve140_v13',
    'HLT_DiPFJetAve160_HFJEC_v16',
    'HLT_DiPFJetAve200_v13',
    'HLT_DiPFJetAve220_HFJEC_v16',
    'HLT_DiPFJetAve260_v14',
    'HLT_DiPFJetAve300_HFJEC_v16',
    'HLT_DiPFJetAve320_v14',
    'HLT_DiPFJetAve400_v14',
    'HLT_DiPFJetAve40_v14',
    'HLT_DiPFJetAve500_v14',
    'HLT_DiPFJetAve60_HFJEC_v15',
    'HLT_DiPFJetAve60_v14',
    'HLT_DiPFJetAve80_HFJEC_v16',
    'HLT_DiPFJetAve80_v13',
    'HLT_DoublePFJets100_PFBTagDeepCSV_p71_v1',
    'HLT_DoublePFJets100_PFBTagDeepJet_p71_v1',
    'HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepCSV_p71_v1',
    'HLT_DoublePFJets116MaxDeta1p6_DoublePFBTagDeepJet_p71_v1',
    'HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepCSV_p71_v1',
    'HLT_DoublePFJets128MaxDeta1p6_DoublePFBTagDeepJet_p71_v1',
    'HLT_DoublePFJets200_PFBTagDeepCSV_p71_v1',
    'HLT_DoublePFJets200_PFBTagDeepJet_p71_v1',
    'HLT_DoublePFJets350_PFBTagDeepCSV_p71_v1',
    'HLT_DoublePFJets350_PFBTagDeepJet_p71_v1',
    'HLT_DoublePFJets40_PFBTagDeepCSV_p71_v1',
    'HLT_DoublePFJets40_PFBTagDeepJet_p71_v1',
    'HLT_Mu12_DoublePFJets100_PFBTagDeepCSV_p71_v1',
    'HLT_Mu12_DoublePFJets100_PFBTagDeepJet_p71_v1',
    'HLT_Mu12_DoublePFJets200_PFBTagDeepCSV_p71_v1',
    'HLT_Mu12_DoublePFJets200_PFBTagDeepJet_p71_v1',
    'HLT_Mu12_DoublePFJets350_PFBTagDeepCSV_p71_v1',
    'HLT_Mu12_DoublePFJets350_PFBTagDeepJet_p71_v1',
    'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71_v1',
    'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepJet_p71_v1',
    'HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71_v1',
    'HLT_Mu12_DoublePFJets40_PFBTagDeepJet_p71_v1',
    'HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepCSV_p71_v1',
    'HLT_Mu12_DoublePFJets54MaxDeta1p6_DoublePFBTagDeepJet_p71_v1',
    'HLT_Mu12eta2p3_PFJet40_v1',
    'HLT_Mu12eta2p3_v1',
    'HLT_PFHT1050_v18',
    'HLT_PFHT180_v17',
    'HLT_PFHT250_v17',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v3',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5_v1',
    'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v9',
    'HLT_PFHT350MinPFJet15_v9',
    'HLT_PFHT350_v19',
    'HLT_PFHT370_v17',
    'HLT_PFHT400_FivePFJet_100_100_60_30_30_DoublePFBTagDeepCSV_4p5_v8',
    'HLT_PFHT400_FivePFJet_100_100_60_30_30_DoublePFBTagDeepJet_4p5_v1',
    'HLT_PFHT400_FivePFJet_100_100_60_30_30_v8',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_DoublePFBTagDeepCSV_4p5_v8',
    'HLT_PFHT400_FivePFJet_120_120_60_30_30_DoublePFBTagDeepJet_4p5_v1',
    'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94_v8',
    'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepJet_2p94_v1',
    'HLT_PFHT400_SixPFJet32_v8',
    'HLT_PFHT430_v17',
    'HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59_v7',
    'HLT_PFHT450_SixPFJet36_PFBTagDeepJet_1p59_v1',
    'HLT_PFHT450_SixPFJet36_v7',
    'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v12',
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v12',
    'HLT_PFHT510_v17',
    'HLT_PFHT590_v17',
    'HLT_PFHT680_v17',
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v12',
    'HLT_PFHT700_PFMET95_PFMHT95_IDTight_v12',
    'HLT_PFHT780_v17',
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v12',
    'HLT_PFHT800_PFMET85_PFMHT85_IDTight_v12',
    'HLT_PFHT890_v17',
    'HLT_PFJet140_v19',
    'HLT_PFJet15_v3',
    'HLT_PFJet200_v19',
    'HLT_PFJet25_v3',
    'HLT_PFJet260_v20',
    'HLT_PFJet320_v20',
    'HLT_PFJet400_v20',
    'HLT_PFJet40_v21',
    'HLT_PFJet450_v21',
    'HLT_PFJet500_v21',
    'HLT_PFJet550_v11',
    'HLT_PFJet60_v21',
    'HLT_PFJet80_v20',
    'HLT_PFJetFwd140_v18',
    'HLT_PFJetFwd15_v3',
    'HLT_PFJetFwd200_v18',
    'HLT_PFJetFwd25_v3',
    'HLT_PFJetFwd260_v19',
    'HLT_PFJetFwd320_v19',
    'HLT_PFJetFwd400_v19',
    'HLT_PFJetFwd40_v19',
    'HLT_PFJetFwd450_v19',
    'HLT_PFJetFwd500_v19',
    'HLT_PFJetFwd60_v19',
    'HLT_PFJetFwd80_v18',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v8',
    'HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v1',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2_v8',
    'HLT_QuadPFJet103_88_75_15_PFBTagDeepJet_1p3_VBF2_v1',
    'HLT_QuadPFJet103_88_75_15_v5',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v8',
    'HLT_QuadPFJet105_88_76_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v1',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2_v8',
    'HLT_QuadPFJet105_88_76_15_PFBTagDeepJet_1p3_VBF2_v1',
    'HLT_QuadPFJet105_88_76_15_v5',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v8',
    'HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v1',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2_v8',
    'HLT_QuadPFJet111_90_80_15_PFBTagDeepJet_1p3_VBF2_v1',
    'HLT_QuadPFJet111_90_80_15_v5',
    'HLT_QuadPFJet70_50_40_30_PFBTagParticleNet_2BTagSum0p65_v1',
    'HLT_QuadPFJet70_50_40_30_v1',
    'HLT_QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65_v1',
    'HLT_QuadPFJet70_50_45_35_PFBTagParticleNet_2BTagSum0p65_v1',
    'HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v8',
    'HLT_QuadPFJet98_83_71_15_DoublePFBTagDeepJet_1p3_7p7_VBF1_v1',
    'HLT_QuadPFJet98_83_71_15_PFBTagDeepCSV_1p3_VBF2_v8',
    'HLT_QuadPFJet98_83_71_15_PFBTagDeepJet_1p3_VBF2_v1',
    'HLT_QuadPFJet98_83_71_15_v5',
    'HLT_Rsq0p35_v15',
    'HLT_Rsq0p40_v15',
    'HLT_RsqMR300_Rsq0p09_MR200_4jet_v15',
    'HLT_RsqMR300_Rsq0p09_MR200_v15',
    'HLT_RsqMR320_Rsq0p09_MR200_4jet_v15',
    'HLT_RsqMR320_Rsq0p09_MR200_v15',
    'HLT_SingleJet30_Mu12_SinglePFJet40_v11'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetMET_selector
streamPhysicsHadronsTaus_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetMET_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetMET_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetMET_selector.triggerConditions = cms.vstring(
    'HLT_CaloMET100_NotCleaned_v4',
    'HLT_CaloMET110_NotCleaned_v4',
    'HLT_CaloMET250_NotCleaned_v4',
    'HLT_CaloMET300_NotCleaned_v4',
    'HLT_CaloMET350_NotCleaned_v4',
    'HLT_CaloMET80_NotCleaned_v4',
    'HLT_CaloMET90_NotCleaned_v4',
    'HLT_CaloMHT90_v4',
    'HLT_DiJet110_35_Mjj650_PFMET110_v9',
    'HLT_DiJet110_35_Mjj650_PFMET120_v9',
    'HLT_DiJet110_35_Mjj650_PFMET130_v9',
    'HLT_L1ETMHadSeeds_v2',
    'HLT_MET105_IsoTrk50_v9',
    'HLT_MET120_IsoTrk50_v9',
    'HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v20',
    'HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v20',
    'HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v19',
    'HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v19',
    'HLT_PFMET100_PFMHT100_IDTight_CaloBTagDeepCSV_3p1_v8',
    'HLT_PFMET100_PFMHT100_IDTight_PFHT60_v9',
    'HLT_PFMET105_IsoTrk50_v1',
    'HLT_PFMET105_PFJet100_looseRecoiling_v1',
    'HLT_PFMET110_PFJet100_looseRecoiling_v1',
    'HLT_PFMET110_PFJet100_v1',
    'HLT_PFMET110_PFMHT110_IDTight_CaloBTagDeepCSV_3p1_v8',
    'HLT_PFMET110_PFMHT110_IDTight_v20',
    'HLT_PFMET120_PFMHT120_IDTight_CaloBTagDeepCSV_3p1_v8',
    'HLT_PFMET120_PFMHT120_IDTight_PFHT60_v9',
    'HLT_PFMET120_PFMHT120_IDTight_v20',
    'HLT_PFMET130_PFMHT130_IDTight_CaloBTagDeepCSV_3p1_v8',
    'HLT_PFMET130_PFMHT130_IDTight_v20',
    'HLT_PFMET140_PFMHT140_IDTight_CaloBTagDeepCSV_3p1_v8',
    'HLT_PFMET140_PFMHT140_IDTight_v20',
    'HLT_PFMET200_BeamHaloCleaned_v9',
    'HLT_PFMET200_NotCleaned_v9',
    'HLT_PFMET250_NotCleaned_v9',
    'HLT_PFMET300_NotCleaned_v9',
    'HLT_PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60_v9',
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF_v1',
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v20',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF_v1',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v9',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v20',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF_v1',
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v19',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF_v1',
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v19',
    'HLT_PFMETTypeOne100_PFMHT100_IDTight_PFHT60_v9',
    'HLT_PFMETTypeOne110_PFMHT110_IDTight_v12',
    'HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60_v9',
    'HLT_PFMETTypeOne120_PFMHT120_IDTight_v12',
    'HLT_PFMETTypeOne130_PFMHT130_IDTight_v12',
    'HLT_PFMETTypeOne140_PFMHT140_IDTight_v11',
    'HLT_PFMETTypeOne200_BeamHaloCleaned_v9',
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v9',
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v9',
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v9'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetTau_selector
streamPhysicsHadronsTaus_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetTau_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetTau_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetTau_selector.triggerConditions = cms.vstring(
    'HLT_DoubleMediumChargedIsoDisplacedPFTauHPS32_Trk1_eta2p1_v1',
    'HLT_DoubleMediumChargedIsoPFTauHPS40_Trk1_eta2p1_v1',
    'HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60_v1',
    'HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75_v1',
    'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1_v1',
    'HLT_DoublePFJets40_Mass500_MediumDeepTauPFTauHPS45_L2NN_MediumDeepTauPFTauHPS20_eta2p1_v1',
    'HLT_DoubleTightChargedIsoPFTauHPS35_Trk1_eta2p1_v1',
    'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v1',
    'HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_v12',
    'HLT_MediumChargedIsoPFTau200HighPtRelaxedIso_Trk50_eta2p1_v12',
    'HLT_MediumChargedIsoPFTau220HighPtRelaxedIso_Trk50_eta2p1_v12',
    'HLT_Photon35_TwoProngs35_v1',
    'HLT_VBF_DoubleLooseChargedIsoPFTauHPS20_Trk1_eta2p1_v1',
    'HLT_VBF_DoubleMediumChargedIsoPFTauHPS20_Trk1_eta2p1_v1',
    'HLT_VBF_DoubleMediumDeepTauPFTauHPS20_eta2p1_v1',
    'HLT_VBF_DoubleTightChargedIsoPFTauHPS20_Trk1_eta2p1_v1'
)


# stream PhysicsMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuon_selector
streamPhysicsMuons_datasetDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuon_selector.triggerConditions = cms.vstring(
    'HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v1',
    'HLT_DoubleL2Mu12NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v1',
    'HLT_DoubleL2Mu14NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v1',
    'HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v2',
    'HLT_DoubleL2Mu23NoVtx_2Cha_v2',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4_v2',
    'HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_v2',
    'HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4_v2',
    'HLT_DoubleL2Mu25NoVtx_2Cha_v2',
    'HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4_v2',
    'HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4_v2',
    'HLT_DoubleL2Mu50_v2',
    'HLT_DoubleL3Iter4Mu16_10NoVtx_DxyMin0p01cm_v1',
    'HLT_DoubleL3Mu16_10NoVtx_DxyMin0p01cm_v1',
    'HLT_DoubleL3Mu18_10NoVtx_DxyMin0p01cm_v1',
    'HLT_DoubleL3Mu20_10NoVtx_DxyMin0p01cm_v1',
    'HLT_DoubleMu3_DCA_PFMET50_PFMHT60_v10',
    'HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v10',
    'HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v10',
    'HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v10',
    'HLT_DoubleMu40NoFiltersNoVtxDisplaced_v1',
    'HLT_DoubleMu43NoFiltersNoVtx_v4',
    'HLT_DoubleMu48NoFiltersNoVtx_v4',
    'HLT_DoubleMu4_Mass3p8_DZ_PFHT350_v8',
    'HLT_L3Iter4Mu10_NoVtx_DxyMin0p01cm_v1',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v5',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v5',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v15',
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v14',
    'HLT_Mu17_TrkIsoVVL_v13',
    'HLT_Mu17_v13',
    'HLT_Mu18_Mu9_DZ_v4',
    'HLT_Mu18_Mu9_SameSign_DZ_v4',
    'HLT_Mu18_Mu9_SameSign_v4',
    'HLT_Mu18_Mu9_v4',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8_v3',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8_v3',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_v3',
    'HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_v3',
    'HLT_Mu19_TrkIsoVVL_v4',
    'HLT_Mu19_v4',
    'HLT_Mu20_Mu10_DZ_v4',
    'HLT_Mu20_Mu10_SameSign_DZ_v4',
    'HLT_Mu20_Mu10_SameSign_v4',
    'HLT_Mu20_Mu10_v4',
    'HLT_Mu23_Mu12_DZ_v4',
    'HLT_Mu23_Mu12_SameSign_DZ_v4',
    'HLT_Mu23_Mu12_SameSign_v4',
    'HLT_Mu23_Mu12_v4',
    'HLT_Mu37_TkMu27_v5',
    'HLT_Mu8_TrkIsoVVL_v12',
    'HLT_Mu8_v12',
    'HLT_TripleMu_10_5_5_DZ_v10',
    'HLT_TripleMu_12_10_5_v10',
    'HLT_TripleMu_5_3_3_Mass3p8_DCA_v3',
    'HLT_TripleMu_5_3_3_Mass3p8_DZ_v8',
    'HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v6',
    'HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx_v12',
    'HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v13'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuonLowMass_selector
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.triggerConditions = cms.vstring(
    'HLT_Dimuon0_Jpsi3p5_Muon2_v5',
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v7',
    'HLT_Dimuon0_Jpsi_L1_NoOS_v7',
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v7',
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v7',
    'HLT_Dimuon0_Jpsi_NoVertexing_v8',
    'HLT_Dimuon0_Jpsi_v8',
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v7',
    'HLT_Dimuon0_LowMass_L1_0er1p5_v8',
    'HLT_Dimuon0_LowMass_L1_4R_v7',
    'HLT_Dimuon0_LowMass_L1_4_v8',
    'HLT_Dimuon0_LowMass_L1_TM530_v6',
    'HLT_Dimuon0_LowMass_v8',
    'HLT_Dimuon0_Upsilon_L1_4p5NoOS_v8',
    'HLT_Dimuon0_Upsilon_L1_4p5_v9',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v7',
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v9',
    'HLT_Dimuon0_Upsilon_L1_5M_v8',
    'HLT_Dimuon0_Upsilon_L1_5_v9',
    'HLT_Dimuon0_Upsilon_Muon_L1_TM0_v6',
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v6',
    'HLT_Dimuon0_Upsilon_NoVertexing_v7',
    'HLT_Dimuon10_PsiPrime_Barrel_Seagulls_v7',
    'HLT_Dimuon12_Upsilon_y1p4_v2',
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v7',
    'HLT_Dimuon18_PsiPrime_noCorrL1_v6',
    'HLT_Dimuon18_PsiPrime_v14',
    'HLT_Dimuon20_Jpsi_Barrel_Seagulls_v7',
    'HLT_Dimuon24_Phi_noCorrL1_v6',
    'HLT_Dimuon24_Upsilon_noCorrL1_v6',
    'HLT_Dimuon25_Jpsi_noCorrL1_v6',
    'HLT_Dimuon25_Jpsi_v14',
    'HLT_DoubleMu2_Jpsi_DoubleTkMu0_Phi_v5',
    'HLT_DoubleMu2_Jpsi_DoubleTrk1_Phi1p05_v6',
    'HLT_DoubleMu3_DoubleEle7p5_CaloIdL_TrackIdL_Upsilon_v4',
    'HLT_DoubleMu3_TkMu_DsTau3Mu_v4',
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v6',
    'HLT_DoubleMu3_Trk_Tau3mu_v12',
    'HLT_DoubleMu4_3_Bs_v15',
    'HLT_DoubleMu4_3_Jpsi_v15',
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v7',
    'HLT_DoubleMu4_Jpsi_Displaced_v7',
    'HLT_DoubleMu4_Jpsi_NoVertexing_v7',
    'HLT_DoubleMu4_MuMuTrk_Displaced_v15',
    'HLT_DoubleMu5_Upsilon_DoubleEle3_CaloIdL_TrackIdL_v4',
    'HLT_Mu20_TkMu0_Phi_v8',
    'HLT_Mu25_TkMu0_Onia_v8',
    'HLT_Mu25_TkMu0_Phi_v8',
    'HLT_Mu30_TkMu0_Psi_v1',
    'HLT_Mu30_TkMu0_Upsilon_v1',
    'HLT_Mu4_L1DoubleMu_v1',
    'HLT_Mu7p5_L2Mu2_Jpsi_v10',
    'HLT_Mu7p5_L2Mu2_Upsilon_v10',
    'HLT_Mu7p5_Track2_Jpsi_v11',
    'HLT_Mu7p5_Track2_Upsilon_v11',
    'HLT_Mu7p5_Track3p5_Jpsi_v11',
    'HLT_Mu7p5_Track3p5_Upsilon_v11',
    'HLT_Mu7p5_Track7_Jpsi_v11',
    'HLT_Mu7p5_Track7_Upsilon_v11',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_Charge1_v4',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_IsoTau15_v4',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_Charge1_v4',
    'HLT_Tau3Mu_Mu7_Mu1_TkMu1_Tau15_v4',
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v5',
    'HLT_TrimuonOpen_5_3p5_2_Upsilon_Muon_v3'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetMuonEG_selector
streamPhysicsMuons_datasetMuonEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetMuonEG_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetMuonEG_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetMuonEG_selector.triggerConditions = cms.vstring(
    'HLT_DiMu4_Ele9_CaloIdL_TrackIdL_DZ_Mass3p8_v17',
    'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ_v17',
    'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v17',
    'HLT_DoubleMu20_7_Mass0to30_L1_DM4EG_v8',
    'HLT_DoubleMu20_7_Mass0to30_L1_DM4_v7',
    'HLT_DoubleMu20_7_Mass0to30_Photon23_v8',
    'HLT_Mu12_DoublePhoton20_v5',
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v15',
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v7',
    'HLT_Mu17_Photon30_IsoCaloId_v6',
    'HLT_Mu20NoFiltersNoVtxDisplaced_Photon20_CaloCustomId_v1',
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v15',
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v7',
    'HLT_Mu27_Ele37_CaloIdL_MW_v5',
    'HLT_Mu37_Ele27_CaloIdL_MW_v5',
    'HLT_Mu38NoFiltersNoVtxDisplaced_Photon38_CaloIdL_v1',
    'HLT_Mu43NoFiltersNoVtxDisplaced_Photon43_CaloIdL_v1',
    'HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v5',
    'HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v5',
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ_v18',
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v18',
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ_v19',
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_v19',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_CaloBtagDeepCSV_1p5_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_CaloDiJet30_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepCSV_1p5_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBtagDeepJet_1p5_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_v1',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v13',
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v11'
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetSingleMuon_selector
streamPhysicsMuons_datasetSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetSingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetSingleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetSingleMuon_selector.triggerConditions = cms.vstring(
    'HLT_CascadeMu100_v3',
    'HLT_HighPtTkMu100_v2',
    'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1_v1',
    'HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_CrossL1_v1',
    'HLT_IsoMu20_eta2p1_TightChargedIsoPFTauHPS27_eta2p1_TightID_CrossL1_v1',
    'HLT_IsoMu20_v15',
    'HLT_IsoMu24_TwoProngs35_v1',
    'HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS180_eta2p1_v1',
    'HLT_IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1_v1',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS20_eta2p1_SingleL1_v1',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_CrossL1_v1',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60_CrossL1_v1',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet75_CrossL1_v1',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS35_L2NN_eta2p1_CrossL1_v1',
    'HLT_IsoMu24_eta2p1_MediumDeepTauPFTauHPS45_L2NN_eta2p1_CrossL1_v1',
    'HLT_IsoMu24_eta2p1_v15',
    'HLT_IsoMu24_v13',
    'HLT_IsoMu27_v16',
    'HLT_IsoMu30_v4',
    'HLT_L1SingleMu18_v3',
    'HLT_L1SingleMu25_v2',
    'HLT_L2Mu10_v7',
    'HLT_L2Mu50_v2',
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v15',
    'HLT_Mu12_v3',
    'HLT_Mu15_IsoVVVL_PFHT450_CaloBTagDeepCSV_4p5_v8',
    'HLT_Mu15_IsoVVVL_PFHT450_PFMET50_v15',
    'HLT_Mu15_IsoVVVL_PFHT450_v15',
    'HLT_Mu15_IsoVVVL_PFHT600_v19',
    'HLT_Mu15_v3',
    'HLT_Mu20_v12',
    'HLT_Mu27_v13',
    'HLT_Mu3_PFJet40_v16',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET100_PFMHT100_IDTight_v2',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET70_PFMHT70_IDTight_v2',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET80_PFMHT80_IDTight_v2',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMET90_PFMHT90_IDTight_v2',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight_v2',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu70_PFMHTNoMu70_IDTight_v2',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu80_PFMHTNoMu80_IDTight_v2',
    'HLT_Mu3er1p5_PFJet100er2p5_PFMETNoMu90_PFMHTNoMu90_IDTight_v2',
    'HLT_Mu4_TrkIsoVVL_DiPFJet90_40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v15',
    'HLT_Mu50_IsoVVVL_PFHT450_v15',
    'HLT_Mu50_v13',
    'HLT_Mu55_v3',
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v16'
)


# stream PhysicsScoutingPFMonitor

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.l1tResults = cms.InputTag('')
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.throw      = cms.bool(False)
streamPhysicsScoutingPFMonitor_datasetScoutingPFMonitor_selector.triggerConditions = cms.vstring(
    'DST_Run3_PFScoutingPixelTracking_v16',
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v14',
    'HLT_Ele35_WPTight_Gsf_v9',
    'HLT_IsoMu27_v16',
    'HLT_Mu50_v13',
    'HLT_PFHT1050_v18',
    'HLT_Photon200_v13'
)


# stream PhysicsZeroBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.throw      = cms.bool(False)
streamPhysicsZeroBias0_datasetEphemeralZeroBias0_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.throw      = cms.bool(False)
streamPhysicsZeroBias0_datasetEphemeralZeroBias1_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')


# stream PhysicsZeroBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.throw      = cms.bool(False)
streamPhysicsZeroBias1_datasetEphemeralZeroBias2_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.throw      = cms.bool(False)
streamPhysicsZeroBias1_datasetEphemeralZeroBias3_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')


# stream PhysicsZeroBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.throw      = cms.bool(False)
streamPhysicsZeroBias2_datasetEphemeralZeroBias4_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.throw      = cms.bool(False)
streamPhysicsZeroBias2_datasetEphemeralZeroBias5_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')


# stream PhysicsZeroBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.throw      = cms.bool(False)
streamPhysicsZeroBias3_datasetEphemeralZeroBias6_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.throw      = cms.bool(False)
streamPhysicsZeroBias3_datasetEphemeralZeroBias7_selector.triggerConditions = cms.vstring('HLT_EphemeralZeroBias_v1')

