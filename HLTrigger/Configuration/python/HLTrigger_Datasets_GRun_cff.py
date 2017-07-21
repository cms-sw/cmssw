# /dev/CMSSW_9_2_0/GRun

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_IsoTrackHB_v1', 
    'HLT_IsoTrackHE_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v12', 
    'HLT_HcalPhiSym_v13')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.triggerConditions = cms.vstring('HLT_HISinglePhoton10_Eta3p1ForPPRef_v6', 
    'HLT_HISinglePhoton20_Eta3p1ForPPRef_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.triggerConditions = cms.vstring('HLT_HISinglePhoton30_Eta3p1ForPPRef_v6', 
    'HLT_HISinglePhoton40_Eta3p1ForPPRef_v6', 
    'HLT_HISinglePhoton50_Eta3p1ForPPRef_v6', 
    'HLT_HISinglePhoton60_Eta3p1ForPPRef_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetIsolatedBunch_selector
streamPhysicsCommissioning_datasetIsolatedBunch_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetIsolatedBunch_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetIsolatedBunch_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetIsolatedBunch_selector.triggerConditions = cms.vstring('HLT_HcalIsolatedbunch_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMonteCarlo_selector
streamPhysicsCommissioning_datasetMonteCarlo_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMonteCarlo_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMonteCarlo_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMonteCarlo_selector.triggerConditions = cms.vstring('MC_AK4CaloJetsFromPV_v3', 
    'MC_AK4CaloJets_v6', 
    'MC_AK4PFJets_v9', 
    'MC_AK8CaloHT_v6', 
    'MC_AK8PFHT_v9', 
    'MC_AK8PFJets_v9', 
    'MC_AK8TrimPFJets_v9', 
    'MC_CaloBTagCSV_v3', 
    'MC_CaloHT_v6', 
    'MC_CaloMET_JetIdCleaned_v6', 
    'MC_CaloMET_v6', 
    'MC_CaloMHT_v6', 
    'MC_Diphoton10_10_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass10_v9', 
    'MC_DoubleEle5_CaloIdL_MW_v10', 
    'MC_DoubleMuNoFiltersNoVtx_v5', 
    'MC_DoubleMu_TrkIsoVVL_DZ_v7', 
    'MC_Ele15_Ele10_CaloIdL_TrackIdL_IsoVL_DZ_v10', 
    'MC_Ele5_WPTight_Gsf_v3', 
    'MC_IsoMu_v10', 
    'MC_PFBTagCSV_v3', 
    'MC_PFHT_v9', 
    'MC_PFMET_v9', 
    'MC_PFMHT_v9', 
    'MC_ReducedIterativeTracking_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring('HLT_L2Mu10_NoVertex_NoBPTX3BX_v4', 
    'HLT_L2Mu10_NoVertex_NoBPTX_v5', 
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v4', 
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v3', 
    'HLT_UncorrectedJetE30_NoBPTX3BX_v3', 
    'HLT_UncorrectedJetE30_NoBPTX_v3', 
    'HLT_UncorrectedJetE60_NoBPTX3BX_v3', 
    'HLT_UncorrectedJetE70_NoBPTX3BX_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_Random_v3', 
    'HLT_ZeroBias_FirstBXAfterTrain_v3', 
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v5', 
    'HLT_ZeroBias_FirstCollisionInTrain_v3', 
    'HLT_ZeroBias_IsolatedBunches_v5', 
    'HLT_ZeroBias_LastCollisionInTrain_v3', 
    'HLT_ZeroBias_v6')


# stream PhysicsEGamma

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetDoubleEG_selector
streamPhysicsEGamma_datasetDoubleEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetDoubleEG_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetDoubleEG_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetDoubleEG_selector.triggerConditions = cms.vstring('HLT_DiEle27_WPTightCaloOnly_L1DoubleEG_v1', 
    'HLT_DiSC30_18_EIso_AND_HE_Mass70_v9', 
    'HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55_v9', 
    'HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v10', 
    'HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55_v9', 
    'HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v10', 
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v9', 
    'HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v9', 
    'HLT_DoubleEle33_CaloIdL_MW_v12', 
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_DZ_PFHT350_v12', 
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT350_v12', 
    'HLT_DoublePhoton33_CaloIdL_v3', 
    'HLT_DoublePhoton70_v3', 
    'HLT_DoublePhoton85_v11', 
    'HLT_ECALHT800_v8', 
    'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v4', 
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v13', 
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v13', 
    'HLT_Ele28_HighEta_SC20_Mass55_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetSingleElectron_selector
streamPhysicsEGamma_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetSingleElectron_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele115_CaloIdVT_GsfTrkIdT_v9', 
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v10', 
    'HLT_Ele135_CaloIdVT_GsfTrkIdT_v2', 
    'HLT_Ele145_CaloIdVT_GsfTrkIdT_v3', 
    'HLT_Ele15_IsoVVVL_PFHT450_CaloBTagCSV_4p5_v1', 
    'HLT_Ele15_IsoVVVL_PFHT450_PFMET50_v8', 
    'HLT_Ele15_IsoVVVL_PFHT450_v8', 
    'HLT_Ele15_IsoVVVL_PFHT600_v12', 
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v8', 
    'HLT_Ele200_CaloIdVT_GsfTrkIdT_v3', 
    'HLT_Ele20_WPLoose_Gsf_v1', 
    'HLT_Ele20_WPTight_Gsf_v1', 
    'HLT_Ele20_eta2p1_WPLoose_Gsf_v1', 
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v10', 
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v10', 
    'HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1_v5', 
    'HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_TightID_CrossL1_v5', 
    'HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTau30_eta2p1_CrossL1_v5', 
    'HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTau30_eta2p1_TightID_CrossL1_v5', 
    'HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTau30_eta2p1_CrossL1_v5', 
    'HLT_Ele24_eta2p1_WPTight_Gsf_TightChargedIsoPFTau30_eta2p1_TightID_CrossL1_v5', 
    'HLT_Ele250_CaloIdVT_GsfTrkIdT_v8', 
    'HLT_Ele27_WPTight_Gsf_v11', 
    'HLT_Ele28_eta2p1_WPTight_Gsf_HT150_v5', 
    'HLT_Ele300_CaloIdVT_GsfTrkIdT_v8', 
    'HLT_Ele30_eta2p1_WPTight_Gsf_CentralPFJet35_EleCleaned_v5', 
    'HLT_Ele32_WPTight_Gsf_L1DoubleEG_v4', 
    'HLT_Ele35_WPTight_Gsf_v4', 
    'HLT_Ele38_WPTight_Gsf_v4', 
    'HLT_Ele40_WPTight_Gsf_v4', 
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v10', 
    'HLT_Ele50_IsoVVVL_PFHT450_v8', 
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v8', 
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetSinglePhoton_selector
streamPhysicsEGamma_datasetSinglePhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetSinglePhoton_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetSinglePhoton_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetSinglePhoton_selector.triggerConditions = cms.vstring('HLT_Photon120_R9Id90_HE10_IsoM_v11', 
    'HLT_Photon120_v10', 
    'HLT_Photon150_v3', 
    'HLT_Photon165_R9Id90_HE10_IsoM_v12', 
    'HLT_Photon175_v11', 
    'HLT_Photon200_v10', 
    'HLT_Photon20_HoverELoose_v8', 
    'HLT_Photon300_NoHE_v10', 
    'HLT_Photon30_HoverELoose_v8', 
    'HLT_Photon33_v3', 
    'HLT_Photon40_HoverELoose_v8', 
    'HLT_Photon50_HoverELoose_v8', 
    'HLT_Photon50_R9Id90_HE10_IsoM_v11', 
    'HLT_Photon50_v10', 
    'HLT_Photon60_HoverELoose_v8', 
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15_v4', 
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_v2', 
    'HLT_Photon60_R9Id90_CaloIdL_IsoL_v2', 
    'HLT_Photon75_R9Id90_HE10_IsoM_v11', 
    'HLT_Photon75_v10', 
    'HLT_Photon90_CaloIdL_PFHT700_v9', 
    'HLT_Photon90_R9Id90_HE10_IsoM_v11', 
    'HLT_Photon90_v10')


# stream PhysicsEndOfFill

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetEmptyBX_selector
streamPhysicsEndOfFill_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetEmptyBX_selector.triggerConditions = cms.vstring('HLT_L1NotBptxOR_v3', 
    'HLT_L1UnpairedBunchBptxMinus_v2', 
    'HLT_L1UnpairedBunchBptxPlus_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetFSQJet1_selector
streamPhysicsEndOfFill_datasetFSQJet1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetFSQJet1_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetFSQJet1_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetFSQJet1_selector.triggerConditions = cms.vstring('HLT_DiPFJet15_NoCaloMatched_v9', 
    'HLT_DiPFJet25_NoCaloMatched_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetFSQJet2_selector
streamPhysicsEndOfFill_datasetFSQJet2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetFSQJet2_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetFSQJet2_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetFSQJet2_selector.triggerConditions = cms.vstring('HLT_DiPFJet15_FBEta3_NoCaloMatched_v10', 
    'HLT_DiPFJet25_FBEta3_NoCaloMatched_v10', 
    'HLT_DiPFJetAve15_HFJEC_v10', 
    'HLT_DiPFJetAve25_HFJEC_v10', 
    'HLT_DiPFJetAve35_HFJEC_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHINCaloJets_selector
streamPhysicsEndOfFill_datasetHINCaloJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHINCaloJets_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHINCaloJets_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHINCaloJets_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet100_v7', 
    'HLT_AK4CaloJet120_v6', 
    'HLT_AK4CaloJet30_v8', 
    'HLT_AK4CaloJet40_v7', 
    'HLT_AK4CaloJet50_v7', 
    'HLT_AK4CaloJet80_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHINPFJets_selector
streamPhysicsEndOfFill_datasetHINPFJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHINPFJets_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHINPFJets_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHINPFJets_selector.triggerConditions = cms.vstring('HLT_AK4PFJet100_v11', 
    'HLT_AK4PFJet120_v10', 
    'HLT_AK4PFJet30_v11', 
    'HLT_AK4PFJet50_v11', 
    'HLT_AK4PFJet80_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.triggerConditions = cms.vstring('HLT_FullTrack_Multiplicity100_v1', 
    'HLT_FullTrack_Multiplicity130_v1', 
    'HLT_FullTrack_Multiplicity155_v2', 
    'HLT_FullTrack_Multiplicity85_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetL1MinimumBias_selector
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_v2')


# stream PhysicsHadronsTaus

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetBTagCSV_selector
streamPhysicsHadronsTaus_datasetBTagCSV_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetBTagCSV_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetBTagCSV_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetBTagCSV_selector.triggerConditions = cms.vstring('HLT_DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33_v1', 
    'HLT_DoublePFJets100_CaloBTagCSV_p33_v1', 
    'HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagCSV_p33_v1', 
    'HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagCSV_p33_v1', 
    'HLT_DoublePFJets200_CaloBTagCSV_p33_v1', 
    'HLT_DoublePFJets350_CaloBTagCSV_p33_v1', 
    'HLT_DoublePFJets40_CaloBTagCSV_p33_v1', 
    'HLT_Mu12_DoublePFJets100_CaloBTagCSV_p33_v1', 
    'HLT_Mu12_DoublePFJets200_CaloBTagCSV_p33_v1', 
    'HLT_Mu12_DoublePFJets350_CaloBTagCSV_p33_v1', 
    'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagCSV_p33_v1', 
    'HLT_Mu12_DoublePFJets40_CaloBTagCSV_p33_v1', 
    'HLT_Mu12_DoublePFJets54MaxDeta1p6_DoubleCaloBTagCSV_p33_v1', 
    'HLT_Mu12_DoublePFJets62MaxDeta1p6_DoubleCaloBTagCSV_p33_v1', 
    'HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0_v1', 
    'HLT_PFHT300PT30_QuadPFJet_75_60_45_40_v1', 
    'HLT_SingleJet30_Mu12_SinglePFJet40_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetBTagMu_selector
streamPhysicsHadronsTaus_datasetBTagMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetBTagMu_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetBTagMu_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetBTagMu_selector.triggerConditions = cms.vstring('HLT_BTagMu_AK4DiJet110_Mu5_v7', 
    'HLT_BTagMu_AK4DiJet170_Mu5_v6', 
    'HLT_BTagMu_AK4DiJet20_Mu5_v7', 
    'HLT_BTagMu_AK4DiJet40_Mu5_v7', 
    'HLT_BTagMu_AK4DiJet70_Mu5_v7', 
    'HLT_BTagMu_AK4Jet300_Mu5_v7', 
    'HLT_BTagMu_AK8Jet300_Mu5_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetDisplacedJet_selector
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.triggerConditions = cms.vstring('HLT_HT400_DisplacedDijet40_DisplacedTrack_v7', 
    'HLT_HT425_v5', 
    'HLT_HT430_DisplacedDijet40_DisplacedTrack_v7', 
    'HLT_HT430_DisplacedDijet60_DisplacedTrack_v7', 
    'HLT_HT430_DisplacedDijet80_DisplacedTrack_v7', 
    'HLT_HT550_DisplacedDijet60_Inclusive_v7', 
    'HLT_HT550_DisplacedDijet80_Inclusive_v5', 
    'HLT_HT650_DisplacedDijet60_Inclusive_v7', 
    'HLT_HT650_DisplacedDijet80_Inclusive_v8', 
    'HLT_HT750_DisplacedDijet80_Inclusive_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetHTMHT_selector
streamPhysicsHadronsTaus_datasetHTMHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetHTMHT_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetHTMHT_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetHTMHT_selector.triggerConditions = cms.vstring('HLT_PFHT500_PFMET100_PFMHT100_IDTight_v4', 
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v4', 
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v4', 
    'HLT_PFHT700_PFMET95_PFMHT95_IDTight_v4', 
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v4', 
    'HLT_PFHT800_PFMET85_PFMHT85_IDTight_v4', 
    'HLT_Rsq0p35_v7', 
    'HLT_Rsq0p40_v7', 
    'HLT_RsqMR300_Rsq0p09_MR200_4jet_v7', 
    'HLT_RsqMR300_Rsq0p09_MR200_v7', 
    'HLT_RsqMR320_Rsq0p09_MR200_4jet_v7', 
    'HLT_RsqMR320_Rsq0p09_MR200_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetJetHT_selector
streamPhysicsHadronsTaus_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetJetHT_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_AK8PFHT750_TrimMass50_v4', 
    'HLT_AK8PFHT800_TrimMass50_v4', 
    'HLT_AK8PFHT850_TrimMass50_v3', 
    'HLT_AK8PFHT900_TrimMass50_v3', 
    'HLT_AK8PFJet140_v7', 
    'HLT_AK8PFJet200_v7', 
    'HLT_AK8PFJet260_v8', 
    'HLT_AK8PFJet320_v8', 
    'HLT_AK8PFJet360_TrimMass30_v10', 
    'HLT_AK8PFJet380_TrimMass30_v3', 
    'HLT_AK8PFJet400_TrimMass30_v4', 
    'HLT_AK8PFJet400_v8', 
    'HLT_AK8PFJet40_v8', 
    'HLT_AK8PFJet420_TrimMass30_v3', 
    'HLT_AK8PFJet450_v8', 
    'HLT_AK8PFJet500_v8', 
    'HLT_AK8PFJet550_v3', 
    'HLT_AK8PFJet60_v7', 
    'HLT_AK8PFJet80_v7', 
    'HLT_AK8PFJetFwd140_v6', 
    'HLT_AK8PFJetFwd200_v6', 
    'HLT_AK8PFJetFwd260_v7', 
    'HLT_AK8PFJetFwd320_v7', 
    'HLT_AK8PFJetFwd400_v7', 
    'HLT_AK8PFJetFwd40_v7', 
    'HLT_AK8PFJetFwd450_v7', 
    'HLT_AK8PFJetFwd500_v7', 
    'HLT_AK8PFJetFwd60_v6', 
    'HLT_AK8PFJetFwd80_v6', 
    'HLT_CaloJet500_NoJetID_v8', 
    'HLT_CaloJet550_NoJetID_v3', 
    'HLT_PFHT1050_v10', 
    'HLT_PFHT180_v9', 
    'HLT_PFHT250_v9', 
    'HLT_PFHT350MinPFJet15_v2', 
    'HLT_PFHT350_v11', 
    'HLT_PFHT370_v9', 
    'HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_2p2_v1', 
    'HLT_PFHT380_SixPFJet32_v1', 
    'HLT_PFHT430_SixPFJet40_PFBTagCSV_1p5_v1', 
    'HLT_PFHT430_SixPFJet40_v3', 
    'HLT_PFHT430_v9', 
    'HLT_PFHT510_v9', 
    'HLT_PFHT590_v9', 
    'HLT_PFHT680_v9', 
    'HLT_PFHT780_v9', 
    'HLT_PFHT890_v9', 
    'HLT_PFJet140_v11', 
    'HLT_PFJet200_v11', 
    'HLT_PFJet260_v12', 
    'HLT_PFJet320_v12', 
    'HLT_PFJet400_v12', 
    'HLT_PFJet40_v13', 
    'HLT_PFJet450_v13', 
    'HLT_PFJet500_v13', 
    'HLT_PFJet550_v3', 
    'HLT_PFJet60_v13', 
    'HLT_PFJet80_v12', 
    'HLT_PFJetFwd140_v10', 
    'HLT_PFJetFwd200_v10', 
    'HLT_PFJetFwd260_v11', 
    'HLT_PFJetFwd320_v11', 
    'HLT_PFJetFwd400_v11', 
    'HLT_PFJetFwd40_v11', 
    'HLT_PFJetFwd450_v11', 
    'HLT_PFJetFwd500_v11', 
    'HLT_PFJetFwd60_v11', 
    'HLT_PFJetFwd80_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetMET_selector
streamPhysicsHadronsTaus_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetMET_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetMET_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetMET_selector.triggerConditions = cms.vstring('HLT_CaloMET100_HBHECleaned_v1', 
    'HLT_CaloMET100_NotCleaned_v1', 
    'HLT_CaloMET110_NotCleaned_v1', 
    'HLT_CaloMET250_HBHECleaned_v1', 
    'HLT_CaloMET250_NotCleaned_v1', 
    'HLT_CaloMET300_HBHECleaned_v1', 
    'HLT_CaloMET350_HBHECleaned_v1', 
    'HLT_CaloMET70_HBHECleaned_v1', 
    'HLT_CaloMET80_HBHECleaned_v1', 
    'HLT_CaloMET80_NotCleaned_v1', 
    'HLT_CaloMET90_HBHECleaned_v1', 
    'HLT_CaloMET90_NotCleaned_v1', 
    'HLT_CaloMHT90_v1', 
    'HLT_DiJet110_35_Mjj650_PFMET110_v1', 
    'HLT_DiJet110_35_Mjj650_PFMET120_v1', 
    'HLT_DiJet110_35_Mjj650_PFMET130_v1', 
    'HLT_L1ETMHadSeeds_v1', 
    'HLT_MET105_IsoTrk50_v2', 
    'HLT_MET120_IsoTrk50_v2', 
    'HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v12', 
    'HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v12', 
    'HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v11', 
    'HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v11', 
    'HLT_PFMET100_PFMHT100_IDTight_CaloBTagCSV_3p1_v1', 
    'HLT_PFMET110_PFMHT110_IDTight_CaloBTagCSV_3p1_v1', 
    'HLT_PFMET110_PFMHT110_IDTight_v12', 
    'HLT_PFMET120_PFMHT120_IDTight_CaloBTagCSV_3p1_v1', 
    'HLT_PFMET120_PFMHT120_IDTight_PFHT60_v1', 
    'HLT_PFMET120_PFMHT120_IDTight_v12', 
    'HLT_PFMET130_PFMHT130_IDTight_CaloBTagCSV_3p1_v1', 
    'HLT_PFMET130_PFMHT130_IDTight_v12', 
    'HLT_PFMET140_PFMHT140_IDTight_CaloBTagCSV_3p1_v1', 
    'HLT_PFMET140_PFMHT140_IDTight_v12', 
    'HLT_PFMET200_HBHECleaned_v1', 
    'HLT_PFMET200_HBHE_BeamHaloCleaned_v1', 
    'HLT_PFMET200_NotCleaned_v1', 
    'HLT_PFMET250_HBHECleaned_v1', 
    'HLT_PFMET300_HBHECleaned_v1', 
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v12', 
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v1', 
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v12', 
    'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v11', 
    'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v11', 
    'HLT_PFMETTypeOne110_PFMHT110_IDTight_v4', 
    'HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60_v1', 
    'HLT_PFMETTypeOne120_PFMHT120_IDTight_v4', 
    'HLT_PFMETTypeOne130_PFMHT130_IDTight_v4', 
    'HLT_PFMETTypeOne140_PFMHT140_IDTight_v3', 
    'HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned_v1', 
    'HLT_TripleJet110_35_35_Mjj650_PFMET110_v1', 
    'HLT_TripleJet110_35_35_Mjj650_PFMET120_v1', 
    'HLT_TripleJet110_35_35_Mjj650_PFMET130_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetTau_selector
streamPhysicsHadronsTaus_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetTau_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetTau_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleLooseChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_v4', 
    'HLT_DoubleLooseChargedIsoPFTau35_Trk1_eta2p1_Reg_v4', 
    'HLT_DoubleLooseChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg_v4', 
    'HLT_DoubleLooseChargedIsoPFTau40_Trk1_eta2p1_Reg_v4', 
    'HLT_DoubleMediumChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_v4', 
    'HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg_v4', 
    'HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg_v4', 
    'HLT_DoubleMediumChargedIsoPFTau40_Trk1_eta2p1_Reg_v4', 
    'HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_v4', 
    'HLT_DoubleTightChargedIsoPFTau35_Trk1_eta2p1_Reg_v4', 
    'HLT_DoubleTightChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg_v4', 
    'HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg_v4', 
    'HLT_IsoMu24_eta2p1_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_v4', 
    'HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr_v4', 
    'HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_v4', 
    'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100_v4', 
    'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET90_v4', 
    'HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_v4', 
    'HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg_v1', 
    'HLT_VBF_DoubleMediumChargedIsoPFTau20_Trk1_eta2p1_Reg_v1', 
    'HLT_VBF_DoubleTightChargedIsoPFTau20_Trk1_eta2p1_Reg_v1')


# stream PhysicsMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetCharmonium_selector
streamPhysicsMuons_datasetCharmonium_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetCharmonium_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetCharmonium_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetCharmonium_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi3p5_Muon2_v2', 
    'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R_v3', 
    'HLT_Dimuon0_Jpsi_L1_NoOS_v3', 
    'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R_v3', 
    'HLT_Dimuon0_Jpsi_NoVertexing_NoOS_v3', 
    'HLT_Dimuon0_Jpsi_NoVertexing_v3', 
    'HLT_Dimuon0_LowMass_L1_0er1p5R_v3', 
    'HLT_Dimuon0_LowMass_L1_0er1p5_v3', 
    'HLT_Dimuon0_LowMass_L1_4R_v3', 
    'HLT_Dimuon0_LowMass_L1_4_v3', 
    'HLT_Dimuon0_LowMass_v3', 
    'HLT_Dimuon10_PsiPrime_Barrel_Seagulls_v3', 
    'HLT_Dimuon18_PsiPrime_noCorrL1_v1', 
    'HLT_Dimuon18_PsiPrime_v10', 
    'HLT_Dimuon20_Jpsi_Barrel_Seagulls_v3', 
    'HLT_Dimuon25_Jpsi_noCorrL1_v1', 
    'HLT_Dimuon25_Jpsi_v10', 
    'HLT_DoubleMu4_3_Bs_v10', 
    'HLT_DoubleMu4_3_Jpsi_Displaced_v11', 
    'HLT_DoubleMu4_JpsiTrkTrk_Displaced_v2', 
    'HLT_DoubleMu4_JpsiTrk_Displaced_v10', 
    'HLT_DoubleMu4_Jpsi_Displaced_v3', 
    'HLT_DoubleMu4_Jpsi_NoVertexing_v3', 
    'HLT_DoubleMu4_PsiPrimeTrk_Displaced_v10', 
    'HLT_Mu7p5_L2Mu2_Jpsi_v7', 
    'HLT_Mu7p5_Track2_Jpsi_v7', 
    'HLT_Mu7p5_Track3p5_Jpsi_v7', 
    'HLT_Mu7p5_Track7_Jpsi_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuon_selector
streamPhysicsMuons_datasetDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuon_selector.triggerConditions = cms.vstring('HLT_DoubleL2Mu50_v1', 
    'HLT_DoubleMu20_7_Mass0to30_Photon23_v3', 
    'HLT_DoubleMu3_DZ_PFMET50_PFMHT60_v2', 
    'HLT_DoubleMu3_DZ_PFMET70_PFMHT70_v2', 
    'HLT_DoubleMu3_DZ_PFMET90_PFMHT90_v2', 
    'HLT_DoubleMu43NoFiltersNoVtx_v2', 
    'HLT_DoubleMu48NoFiltersNoVtx_v2', 
    'HLT_DoubleMu4_Mass8_DZ_PFHT350_v2', 
    'HLT_DoubleMu8_Mass8_PFHT350_v2', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v1', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v1', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v11', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v10', 
    'HLT_Mu17_TrkIsoVVL_v8', 
    'HLT_Mu17_v8', 
    'HLT_Mu8_TrkIsoVVL_v8', 
    'HLT_Mu8_v8', 
    'HLT_TripleMu_10_5_5_DZ_v6', 
    'HLT_TripleMu_12_10_5_v6', 
    'HLT_TripleMu_5_3_3_Mass3p8to60_DZ_v4', 
    'HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v1', 
    'HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx_v7', 
    'HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuonLowMass_selector
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.triggerConditions = cms.vstring('HLT_Dimuon0_LowMass_L1_TM530_v2', 
    'HLT_DoubleMu3_Trk_Tau3mu_NoL1Mass_v2', 
    'HLT_DoubleMu3_Trk_Tau3mu_v7', 
    'HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetMuOnia_selector
streamPhysicsMuons_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetMuOnia_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Upsilon_L1_4p5NoOS_v3', 
    'HLT_Dimuon0_Upsilon_L1_4p5_v3', 
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0M_v3', 
    'HLT_Dimuon0_Upsilon_L1_4p5er2p0_v3', 
    'HLT_Dimuon0_Upsilon_L1_5M_v3', 
    'HLT_Dimuon0_Upsilon_L1_5_v3', 
    'HLT_Dimuon0_Upsilon_Muon_L1_TM0_v2', 
    'HLT_Dimuon0_Upsilon_Muon_NoL1Mass_v2', 
    'HLT_Dimuon0_Upsilon_NoVertexing_v3', 
    'HLT_Dimuon10_Upsilon_Barrel_Seagulls_v3', 
    'HLT_Dimuon12_Upsilon_eta1p5_v10', 
    'HLT_Dimuon14_Phi_Barrel_Seagulls_v3', 
    'HLT_Dimuon24_Phi_noCorrL1_v1', 
    'HLT_Dimuon24_Upsilon_noCorrL1_v1', 
    'HLT_Mu20_TkMu0_Phi_v3', 
    'HLT_Mu25_TkMu0_Onia_v3', 
    'HLT_Mu25_TkMu0_Phi_v3', 
    'HLT_Mu30_TkMu0_Onia_v3', 
    'HLT_Mu7p5_L2Mu2_Upsilon_v7', 
    'HLT_Mu7p5_Track2_Upsilon_v7', 
    'HLT_Mu7p5_Track3p5_Upsilon_v7', 
    'HLT_Mu7p5_Track7_Upsilon_v7', 
    'HLT_Trimuon5_3p5_2_Upsilon_Muon_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetMuonEG_selector
streamPhysicsMuons_datasetMuonEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetMuonEG_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetMuonEG_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetMuonEG_selector.triggerConditions = cms.vstring('HLT_DiMu9_Ele9_CaloIdL_TrackIdL_DZ_v11', 
    'HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v11', 
    'HLT_DoubleMu20_7_Mass0to30_L1_DM4EG_v3', 
    'HLT_DoubleMu20_7_Mass0to30_L1_DM4_v3', 
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v9', 
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v1', 
    'HLT_Mu17_Photon30_IsoCaloId_v1', 
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v9', 
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v1', 
    'HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v2', 
    'HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v2', 
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_DZ_v12', 
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v12', 
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_DZ_v11', 
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT350_v11', 
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v7', 
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetSingleMuon_selector
streamPhysicsMuons_datasetSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetSingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetSingleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetSingleMuon_selector.triggerConditions = cms.vstring('HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1_v4', 
    'HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_TightID_CrossL1_v4', 
    'HLT_IsoMu20_eta2p1_MediumChargedIsoPFTau27_eta2p1_CrossL1_v4', 
    'HLT_IsoMu20_eta2p1_MediumChargedIsoPFTau27_eta2p1_TightID_CrossL1_v4', 
    'HLT_IsoMu20_eta2p1_TightChargedIsoPFTau27_eta2p1_CrossL1_v4', 
    'HLT_IsoMu20_eta2p1_TightChargedIsoPFTau27_eta2p1_TightID_CrossL1_v4', 
    'HLT_IsoMu20_v10', 
    'HLT_IsoMu24_eta2p1_LooseChargedIsoPFTau20_SingleL1_v4', 
    'HLT_IsoMu24_eta2p1_LooseChargedIsoPFTau20_TightID_SingleL1_v4', 
    'HLT_IsoMu24_eta2p1_LooseChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_CrossL1_v4', 
    'HLT_IsoMu24_eta2p1_LooseChargedIsoPFTau35_Trk1_eta2p1_Reg_CrossL1_v4', 
    'HLT_IsoMu24_eta2p1_MediumChargedIsoPFTau20_SingleL1_v4', 
    'HLT_IsoMu24_eta2p1_MediumChargedIsoPFTau20_TightID_SingleL1_v4', 
    'HLT_IsoMu24_eta2p1_MediumChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_CrossL1_v4', 
    'HLT_IsoMu24_eta2p1_MediumChargedIsoPFTau35_Trk1_eta2p1_Reg_CrossL1_v4', 
    'HLT_IsoMu24_eta2p1_TightChargedIsoPFTau20_SingleL1_v4', 
    'HLT_IsoMu24_eta2p1_TightChargedIsoPFTau20_TightID_SingleL1_v4', 
    'HLT_IsoMu24_eta2p1_TightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_CrossL1_v4', 
    'HLT_IsoMu24_eta2p1_TightChargedIsoPFTau35_Trk1_eta2p1_Reg_CrossL1_v4', 
    'HLT_IsoMu24_eta2p1_v10', 
    'HLT_IsoMu24_v8', 
    'HLT_IsoMu27_v11', 
    'HLT_L1SingleMu18_v3', 
    'HLT_L1SingleMu25_v2', 
    'HLT_L2Mu10_v5', 
    'HLT_L2Mu50_v1', 
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v6', 
    'HLT_Mu15_IsoVVVL_PFHT450_CaloBTagCSV_4p5_v1', 
    'HLT_Mu15_IsoVVVL_PFHT450_PFMET50_v7', 
    'HLT_Mu15_IsoVVVL_PFHT450_v7', 
    'HLT_Mu15_IsoVVVL_PFHT600_v11', 
    'HLT_Mu20_v8', 
    'HLT_Mu27_v9', 
    'HLT_Mu3_PFJet40_v9', 
    'HLT_Mu50_IsoVVVL_PFHT450_v7', 
    'HLT_Mu50_v9', 
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v7', 
    'HLT_OldMu100_v1', 
    'HLT_TkMu100_v1')

