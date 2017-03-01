# /dev/CMSSW_9_0_0/GRun

import FWCore.ParameterSet.Config as cms


# stream Parking

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParking_datasetParkingHT430to450_selector
streamParking_datasetParkingHT430to450_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParking_datasetParkingHT430to450_selector.l1tResults = cms.InputTag('')
streamParking_datasetParkingHT430to450_selector.throw      = cms.bool(False)
streamParking_datasetParkingHT430to450_selector.triggerConditions = cms.vstring('HLT_HT430to450_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParking_datasetParkingHT450to470_selector
streamParking_datasetParkingHT450to470_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParking_datasetParkingHT450to470_selector.l1tResults = cms.InputTag('')
streamParking_datasetParkingHT450to470_selector.throw      = cms.bool(False)
streamParking_datasetParkingHT450to470_selector.triggerConditions = cms.vstring('HLT_HT450to470_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParking_datasetParkingHT470to500_selector
streamParking_datasetParkingHT470to500_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParking_datasetParkingHT470to500_selector.l1tResults = cms.InputTag('')
streamParking_datasetParkingHT470to500_selector.throw      = cms.bool(False)
streamParking_datasetParkingHT470to500_selector.triggerConditions = cms.vstring('HLT_HT470to500_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParking_datasetParkingHT500to550_selector
streamParking_datasetParkingHT500to550_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParking_datasetParkingHT500to550_selector.l1tResults = cms.InputTag('')
streamParking_datasetParkingHT500to550_selector.throw      = cms.bool(False)
streamParking_datasetParkingHT500to550_selector.triggerConditions = cms.vstring('HLT_HT500to550_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParking_datasetParkingHT550to650_selector
streamParking_datasetParkingHT550to650_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParking_datasetParkingHT550to650_selector.l1tResults = cms.InputTag('')
streamParking_datasetParkingHT550to650_selector.throw      = cms.bool(False)
streamParking_datasetParkingHT550to650_selector.triggerConditions = cms.vstring('HLT_HT550to650_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParking_datasetParkingHT650_selector
streamParking_datasetParkingHT650_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParking_datasetParkingHT650_selector.l1tResults = cms.InputTag('')
streamParking_datasetParkingHT650_selector.throw      = cms.bool(False)
streamParking_datasetParkingHT650_selector.triggerConditions = cms.vstring('HLT_HT650_v5')


# stream ParkingHLTPhysics

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingHLTPhysics_datasetHLTPhysics0_selector
streamParkingHLTPhysics_datasetHLTPhysics0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingHLTPhysics_datasetHLTPhysics0_selector.l1tResults = cms.InputTag('')
streamParkingHLTPhysics_datasetHLTPhysics0_selector.throw      = cms.bool(False)
streamParkingHLTPhysics_datasetHLTPhysics0_selector.triggerConditions = cms.vstring('HLT_L1FatEvents_part0_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingHLTPhysics_datasetHLTPhysics1_selector
streamParkingHLTPhysics_datasetHLTPhysics1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingHLTPhysics_datasetHLTPhysics1_selector.l1tResults = cms.InputTag('')
streamParkingHLTPhysics_datasetHLTPhysics1_selector.throw      = cms.bool(False)
streamParkingHLTPhysics_datasetHLTPhysics1_selector.triggerConditions = cms.vstring('HLT_L1FatEvents_part1_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingHLTPhysics_datasetHLTPhysics2_selector
streamParkingHLTPhysics_datasetHLTPhysics2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingHLTPhysics_datasetHLTPhysics2_selector.l1tResults = cms.InputTag('')
streamParkingHLTPhysics_datasetHLTPhysics2_selector.throw      = cms.bool(False)
streamParkingHLTPhysics_datasetHLTPhysics2_selector.triggerConditions = cms.vstring('HLT_L1FatEvents_part2_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamParkingHLTPhysics_datasetHLTPhysics3_selector
streamParkingHLTPhysics_datasetHLTPhysics3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamParkingHLTPhysics_datasetHLTPhysics3_selector.l1tResults = cms.InputTag('')
streamParkingHLTPhysics_datasetHLTPhysics3_selector.throw      = cms.bool(False)
streamParkingHLTPhysics_datasetHLTPhysics3_selector.triggerConditions = cms.vstring('HLT_L1FatEvents_part3_v1')


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetCommissioning_selector
streamPhysicsCommissioning_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetCommissioning_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_DiSC30_18_EIso_AND_HE_Mass70_v6', 
    'HLT_HcalIsolatedbunch_v2', 
    'HLT_IsoTrackHB_v3', 
    'HLT_IsoTrackHE_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_L1FatEvents_v2', 
    'HLT_Physics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalHPDNoise_selector
streamPhysicsCommissioning_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v10', 
    'HLT_HcalPhiSym_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMonteCarlo_selector
streamPhysicsCommissioning_datasetMonteCarlo_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMonteCarlo_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMonteCarlo_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMonteCarlo_selector.triggerConditions = cms.vstring('MC_AK4CaloJets_v3', 
    'MC_AK4PFJets_v6', 
    'MC_AK8CaloHT_v3', 
    'MC_AK8PFHT_v6', 
    'MC_AK8PFJets_v6', 
    'MC_AK8TrimPFJets_v6', 
    'MC_CaloHT_v3', 
    'MC_CaloMET_JetIdCleaned_v3', 
    'MC_CaloMET_v3', 
    'MC_CaloMHT_v3', 
    'MC_Diphoton10_10_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass10_v6', 
    'MC_DoubleEle5_CaloIdL_GsfTrkIdVL_MW_v6', 
    'MC_DoubleGlbTrkMu_TrkIsoVVL_DZ_v4', 
    'MC_DoubleL1Tau_MediumIsoPFTau32_Trk1_eta2p1_Reg_v6', 
    'MC_DoubleMuNoFiltersNoVtx_v3', 
    'MC_DoubleMu_TrkIsoVVL_DZ_v4', 
    'MC_Ele15_Ele10_CaloIdL_TrackIdL_IsoVL_DZ_v7', 
    'MC_Ele5_WPLoose_Gsf_v8', 
    'MC_IsoMu_v7', 
    'MC_IsoTkMu15_v6', 
    'MC_LooseIsoPFTau20_v5', 
    'MC_LooseIsoPFTau50_Trk30_eta2p1_v4', 
    'MC_PFHT_v6', 
    'MC_PFMET_v6', 
    'MC_PFMHT_v6', 
    'MC_ReducedIterativeTracking_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetNoBPTX_selector
streamPhysicsCommissioning_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetNoBPTX_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetNoBPTX_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_v4', 
    'HLT_JetE30_NoBPTX_v4', 
    'HLT_JetE50_NoBPTX3BX_v4', 
    'HLT_JetE70_NoBPTX3BX_v4', 
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v2', 
    'HLT_L2Mu10_NoVertex_NoBPTX_v3', 
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v2', 
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_Random_v2', 
    'HLT_ZeroBias_FirstBXAfterTrain_v1', 
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_copy_v1', 
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v3', 
    'HLT_ZeroBias_FirstCollisionInTrain_v1', 
    'HLT_ZeroBias_IsolatedBunches_v3', 
    'HLT_ZeroBias_v4')


# stream PhysicsEGamma

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetDoubleEG_selector
streamPhysicsEGamma_datasetDoubleEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetDoubleEG_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetDoubleEG_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetDoubleEG_selector.triggerConditions = cms.vstring('HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_DoublePixelVeto_Mass55_v7', 
    'HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_DoublePixelVeto_Mass55_v7', 
    'HLT_Diphoton30_18_R9Id_OR_IsoCaloId_AND_HE_R9Id_DoublePixelSeedMatch_Mass70_v7', 
    'HLT_Diphoton30_18_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v7', 
    'HLT_Diphoton30_18_Solid_R9Id_AND_IsoCaloId_AND_HE_R9Id_Mass55_v7', 
    'HLT_DoubleEle24_22_eta2p1_WPLoose_Gsf_v8', 
    'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_MW_v10', 
    'HLT_DoubleEle33_CaloIdL_MW_v8', 
    'HLT_DoubleEle33_CaloIdL_v7', 
    'HLT_DoubleEle37_Ele27_CaloIdL_GsfTrkIdVL_v7', 
    'HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300_v10', 
    'HLT_DoublePhoton60_v7', 
    'HLT_DoublePhoton85_v8', 
    'HLT_ECALHT800_v6', 
    'HLT_Ele10_CaloIdM_TrackIdM_CentralPFJet30_BTagCSV_p13_v7', 
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v9', 
    'HLT_Ele12_CaloIdL_TrackIdL_IsoVL_v8', 
    'HLT_Ele12_CaloIdM_TrackIdM_PFJet30_v9', 
    'HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v8', 
    'HLT_Ele17_CaloIdL_GsfTrkIdVL_v6', 
    'HLT_Ele17_CaloIdL_TrackIdL_IsoVL_PFJet30_v7', 
    'HLT_Ele17_CaloIdL_TrackIdL_IsoVL_v7', 
    'HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v7', 
    'HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v9', 
    'HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_v9', 
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v9', 
    'HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v8', 
    'HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v9', 
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_L1JetTauSeeded_v3', 
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v9', 
    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v9', 
    'HLT_Ele27_HighEta_Ele20_Mass55_v8', 
    'HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v7', 
    'HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v9', 
    'HLT_Photon26_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon16_AND_HE10_R9Id65_Eta2_Mass60_v9', 
    'HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon22_AND_HE10_R9Id65_Eta2_Mass15_v9', 
    'HLT_Photon42_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon25_AND_HE10_R9Id65_Eta2_Mass15_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetSingleElectron_selector
streamPhysicsEGamma_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetSingleElectron_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele105_CaloIdVT_GsfTrkIdT_v8', 
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v7', 
    'HLT_Ele145_CaloIdVT_GsfTrkIdT_v1', 
    'HLT_Ele15_IsoVVVL_BTagCSV_p067_PFHT400_v7', 
    'HLT_Ele15_IsoVVVL_PFHT400_PFMET50_v6', 
    'HLT_Ele15_IsoVVVL_PFHT400_v6', 
    'HLT_Ele15_IsoVVVL_PFHT600_v9', 
    'HLT_Ele200_CaloIdVT_GsfTrkIdT_v1', 
    'HLT_Ele20_eta2p1_WPLoose_Gsf_LooseIsoPFTau28_v3', 
    'HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau29_v3', 
    'HLT_Ele22_eta2p1_WPLoose_Gsf_v9', 
    'HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30_v4', 
    'HLT_Ele250_CaloIdVT_GsfTrkIdT_v6', 
    'HLT_Ele25_WPTight_Gsf_v7', 
    'HLT_Ele25_eta2p1_WPTight_Gsf_v7', 
    'HLT_Ele27_WPLoose_Gsf_WHbbBoost_v9', 
    'HLT_Ele27_WPTight_Gsf_L1JetTauSeeded_v4', 
    'HLT_Ele27_WPTight_Gsf_v7', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_HT200_v10', 
    'HLT_Ele27_eta2p1_WPLoose_Gsf_v8', 
    'HLT_Ele27_eta2p1_WPTight_Gsf_v8', 
    'HLT_Ele300_CaloIdVT_GsfTrkIdT_v6', 
    'HLT_Ele30_WPTight_Gsf_v1', 
    'HLT_Ele30_eta2p1_WPTight_Gsf_v1', 
    'HLT_Ele32_WPTight_Gsf_v1', 
    'HLT_Ele32_eta2p1_WPTight_Gsf_v8', 
    'HLT_Ele36_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1_v3', 
    'HLT_Ele45_CaloIdVT_GsfTrkIdT_PFJet200_PFJet50_v9', 
    'HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v7', 
    'HLT_Ele50_IsoVVVL_PFHT400_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEGamma_datasetSinglePhoton_selector
streamPhysicsEGamma_datasetSinglePhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEGamma_datasetSinglePhoton_selector.l1tResults = cms.InputTag('')
streamPhysicsEGamma_datasetSinglePhoton_selector.throw      = cms.bool(False)
streamPhysicsEGamma_datasetSinglePhoton_selector.triggerConditions = cms.vstring('HLT_Photon120_R9Id90_HE10_Iso40_EBOnly_PFMET40_v9', 
    'HLT_Photon120_R9Id90_HE10_Iso40_EBOnly_VBF_v7', 
    'HLT_Photon120_R9Id90_HE10_IsoM_v8', 
    'HLT_Photon120_v7', 
    'HLT_Photon135_PFMET100_v8', 
    'HLT_Photon165_HE10_v8', 
    'HLT_Photon165_R9Id90_HE10_IsoM_v9', 
    'HLT_Photon175_v8', 
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_PFMET40_v9', 
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_VBF_v7', 
    'HLT_Photon22_R9Id90_HE10_IsoM_v7', 
    'HLT_Photon22_v6', 
    'HLT_Photon250_NoHE_v7', 
    'HLT_Photon300_NoHE_v7', 
    'HLT_Photon30_R9Id90_HE10_IsoM_v8', 
    'HLT_Photon30_v7', 
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_PFMET40_v9', 
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_VBF_v8', 
    'HLT_Photon36_R9Id90_HE10_IsoM_v8', 
    'HLT_Photon36_v7', 
    'HLT_Photon500_v6', 
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_PFMET40_v9', 
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_VBF_v7', 
    'HLT_Photon50_R9Id90_HE10_IsoM_v8', 
    'HLT_Photon50_v7', 
    'HLT_Photon600_v6', 
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_PFMET40_v9', 
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_VBF_v7', 
    'HLT_Photon75_R9Id90_HE10_IsoM_v8', 
    'HLT_Photon75_v7', 
    'HLT_Photon90_CaloIdL_PFHT600_v8', 
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_PFMET40_v9', 
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_VBF_v7', 
    'HLT_Photon90_R9Id90_HE10_IsoM_v8', 
    'HLT_Photon90_v7')


# stream PhysicsEndOfFill

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetEmptyBX_selector
streamPhysicsEndOfFill_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetEmptyBX_selector.triggerConditions = cms.vstring('HLT_L1BptxMinus_v2', 
    'HLT_L1BptxPlus_v2', 
    'HLT_L1NotBptxOR_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetFSQJets_selector
streamPhysicsEndOfFill_datasetFSQJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetFSQJets_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetFSQJets_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetFSQJets_selector.triggerConditions = cms.vstring('HLT_DiPFJet15_FBEta3_NoCaloMatched_v6', 
    'HLT_DiPFJet15_NoCaloMatched_v5', 
    'HLT_DiPFJet25_FBEta3_NoCaloMatched_v6', 
    'HLT_DiPFJet25_NoCaloMatched_v5', 
    'HLT_DiPFJetAve15_HFJEC_v6', 
    'HLT_DiPFJetAve25_HFJEC_v6', 
    'HLT_DiPFJetAve35_HFJEC_v6', 
    'HLT_PFJet15_NoCaloMatched_v7', 
    'HLT_PFJet25_NoCaloMatched_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHINCaloJets_selector
streamPhysicsEndOfFill_datasetHINCaloJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHINCaloJets_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHINCaloJets_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHINCaloJets_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet100_v4', 
    'HLT_AK4CaloJet30_v5', 
    'HLT_AK4CaloJet40_v4', 
    'HLT_AK4CaloJet50_v4', 
    'HLT_AK4CaloJet80_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHINPFJets_selector
streamPhysicsEndOfFill_datasetHINPFJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHINPFJets_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHINPFJets_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHINPFJets_selector.triggerConditions = cms.vstring('HLT_AK4PFJet100_v7', 
    'HLT_AK4PFJet30_v7', 
    'HLT_AK4PFJet50_v7', 
    'HLT_AK4PFJet80_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHINPhoton_selector
streamPhysicsEndOfFill_datasetHINPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHINPhoton_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHINPhoton_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHINPhoton_selector.triggerConditions = cms.vstring('HLT_HISinglePhoton10_v4', 
    'HLT_HISinglePhoton15_v4', 
    'HLT_HISinglePhoton20_v4', 
    'HLT_HISinglePhoton40_v4', 
    'HLT_HISinglePhoton60_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetHighMultiplicityEOF_selector.triggerConditions = cms.vstring('HLT_FullTracks_Multiplicity100_v5', 
    'HLT_FullTracks_Multiplicity130_v5', 
    'HLT_FullTracks_Multiplicity150_v5', 
    'HLT_FullTracks_Multiplicity80_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsEndOfFill_datasetL1MinimumBias_selector
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.l1tResults = cms.InputTag('')
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.throw      = cms.bool(False)
streamPhysicsEndOfFill_datasetL1MinimumBias_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_AND_v2')


# stream PhysicsHadronsTaus

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetBTagCSV_selector
streamPhysicsHadronsTaus_datasetBTagCSV_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetBTagCSV_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetBTagCSV_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetBTagCSV_selector.triggerConditions = cms.vstring('HLT_DoubleJet90_Double30_DoubleBTagCSV_p087_v5', 
    'HLT_DoubleJet90_Double30_TripleBTagCSV_p087_v5', 
    'HLT_DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6_v5', 
    'HLT_DoubleJetsC100_DoubleBTagCSV_p026_DoublePFJetsC160_v5', 
    'HLT_DoubleJetsC100_SingleBTagCSV_p014_SinglePFJetC350_v5', 
    'HLT_DoubleJetsC100_SingleBTagCSV_p014_v3', 
    'HLT_DoubleJetsC100_SingleBTagCSV_p026_SinglePFJetC350_v5', 
    'HLT_DoubleJetsC100_SingleBTagCSV_p026_v3', 
    'HLT_DoubleJetsC112_DoubleBTagCSV_p014_DoublePFJetsC112MaxDeta1p6_v5', 
    'HLT_DoubleJetsC112_DoubleBTagCSV_p026_DoublePFJetsC172_v5', 
    'HLT_QuadJet45_DoubleBTagCSV_p087_v6', 
    'HLT_QuadJet45_TripleBTagCSV_p087_v6', 
    'HLT_QuadPFJet_BTagCSV_p016_VBF_Mqq460_v5', 
    'HLT_QuadPFJet_BTagCSV_p016_VBF_Mqq500_v5', 
    'HLT_QuadPFJet_BTagCSV_p016_p11_VBF_Mqq200_v5', 
    'HLT_QuadPFJet_BTagCSV_p016_p11_VBF_Mqq240_v5', 
    'HLT_Rsq0p02_MR400_TriPFJet80_60_40_DoubleBTagCSV_p063_Mbb60_200_v3', 
    'HLT_Rsq0p02_MR450_TriPFJet80_60_40_DoubleBTagCSV_p063_Mbb60_200_v3', 
    'HLT_Rsq0p02_MR500_TriPFJet80_60_40_DoubleBTagCSV_p063_Mbb60_200_v2', 
    'HLT_Rsq0p02_MR550_TriPFJet80_60_40_DoubleBTagCSV_p063_Mbb60_200_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetBTagMu_selector
streamPhysicsHadronsTaus_datasetBTagMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetBTagMu_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetBTagMu_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetBTagMu_selector.triggerConditions = cms.vstring('HLT_BTagMu_AK8Jet300_Mu5_v4', 
    'HLT_BTagMu_DiJet110_Mu5_v5', 
    'HLT_BTagMu_DiJet170_Mu5_v4', 
    'HLT_BTagMu_DiJet20_Mu5_v5', 
    'HLT_BTagMu_DiJet40_Mu5_v5', 
    'HLT_BTagMu_DiJet70_Mu5_v5', 
    'HLT_BTagMu_Jet300_Mu5_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetDisplacedJet_selector
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetDisplacedJet_selector.triggerConditions = cms.vstring('HLT_HT200_v4', 
    'HLT_HT250_DisplacedDijet40_DisplacedTrack_v5', 
    'HLT_HT275_v4', 
    'HLT_HT325_v4', 
    'HLT_HT350_DisplacedDijet40_DisplacedTrack_v5', 
    'HLT_HT350_DisplacedDijet40_Inclusive_v4', 
    'HLT_HT350_DisplacedDijet80_DisplacedTrack_v5', 
    'HLT_HT350_DisplacedDijet80_Tight_DisplacedTrack_v5', 
    'HLT_HT425_v4', 
    'HLT_HT550_DisplacedDijet80_Inclusive_v2', 
    'HLT_HT575_v4', 
    'HLT_HT650_DisplacedDijet80_Inclusive_v5', 
    'HLT_HT750_DisplacedDijet80_Inclusive_v5', 
    'HLT_VBF_DisplacedJet40_DisplacedTrack_2TrackIP2DSig5_v5', 
    'HLT_VBF_DisplacedJet40_DisplacedTrack_v5', 
    'HLT_VBF_DisplacedJet40_TightID_DisplacedTrack_v5', 
    'HLT_VBF_DisplacedJet40_TightID_Hadronic_v5', 
    'HLT_VBF_DisplacedJet40_VTightID_DisplacedTrack_v5', 
    'HLT_VBF_DisplacedJet40_VTightID_Hadronic_v5', 
    'HLT_VBF_DisplacedJet40_VVTightID_DisplacedTrack_v5', 
    'HLT_VBF_DisplacedJet40_VVTightID_Hadronic_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetHTMHT_selector
streamPhysicsHadronsTaus_datasetHTMHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetHTMHT_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetHTMHT_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetHTMHT_selector.triggerConditions = cms.vstring('HLT_DiPFJet40_DEta3p5_MJJ600_PFMETNoMu140_v6', 
    'HLT_DiPFJet40_DEta3p5_MJJ600_PFMETNoMu80_v5', 
    'HLT_PFHT200_DiPFJetAve90_PFAlphaT0p63_v8', 
    'HLT_PFHT200_PFAlphaT0p51_v8', 
    'HLT_PFHT250_DiPFJetAve90_PFAlphaT0p58_v8', 
    'HLT_PFHT300_DiPFJetAve90_PFAlphaT0p54_v8', 
    'HLT_PFHT300_PFMET110_v6', 
    'HLT_PFHT350_DiPFJetAve90_PFAlphaT0p53_v8', 
    'HLT_PFHT400_DiPFJetAve90_PFAlphaT0p52_v8', 
    'HLT_Rsq0p25_v6', 
    'HLT_Rsq0p30_v6', 
    'HLT_RsqMR270_Rsq0p09_MR200_4jet_v6', 
    'HLT_RsqMR270_Rsq0p09_MR200_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetJetHT_selector
streamPhysicsHadronsTaus_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetJetHT_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_AK8DiPFJet250_200_TrimMass30_BTagCSV_p20_v5', 
    'HLT_AK8DiPFJet250_200_TrimMass30_v5', 
    'HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p087_v1', 
    'HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20_v5', 
    'HLT_AK8DiPFJet280_200_TrimMass30_v5', 
    'HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p087_v1', 
    'HLT_AK8DiPFJet300_200_TrimMass30_BTagCSV_p20_v1', 
    'HLT_AK8DiPFJet300_200_TrimMass30_v1', 
    'HLT_AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20_v6', 
    'HLT_AK8PFHT650_TrimR0p1PT0p03Mass50_v7', 
    'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v8', 
    'HLT_AK8PFHT750_TrimMass50_v1', 
    'HLT_AK8PFHT800_TrimMass50_v1', 
    'HLT_AK8PFJet140_v4', 
    'HLT_AK8PFJet200_v4', 
    'HLT_AK8PFJet260_v5', 
    'HLT_AK8PFJet320_v5', 
    'HLT_AK8PFJet360_TrimMass30_v7', 
    'HLT_AK8PFJet400_TrimMass30_v1', 
    'HLT_AK8PFJet400_v5', 
    'HLT_AK8PFJet40_v5', 
    'HLT_AK8PFJet450_v5', 
    'HLT_AK8PFJet500_v5', 
    'HLT_AK8PFJet60_v4', 
    'HLT_AK8PFJet80_v4', 
    'HLT_CaloJet500_NoJetID_v5', 
    'HLT_DiCentralPFJet170_CFMax0p1_v5', 
    'HLT_DiCentralPFJet170_v5', 
    'HLT_DiCentralPFJet330_CFMax0p5_v5', 
    'HLT_DiCentralPFJet430_v5', 
    'HLT_DiJetVBFMu_PassThrough_v1', 
    'HLT_DiJetVBF_PassThrough_v1', 
    'HLT_DiPFJetAve100_HFJEC_v8', 
    'HLT_DiPFJetAve140_v7', 
    'HLT_DiPFJetAve160_HFJEC_v8', 
    'HLT_DiPFJetAve200_v7', 
    'HLT_DiPFJetAve220_HFJEC_v9', 
    'HLT_DiPFJetAve260_v8', 
    'HLT_DiPFJetAve300_HFJEC_v9', 
    'HLT_DiPFJetAve320_v8', 
    'HLT_DiPFJetAve400_v8', 
    'HLT_DiPFJetAve40_v8', 
    'HLT_DiPFJetAve500_v8', 
    'HLT_DiPFJetAve60_HFJEC_v8', 
    'HLT_DiPFJetAve60_v8', 
    'HLT_DiPFJetAve80_HFJEC_v8', 
    'HLT_DiPFJetAve80_v7', 
    'HLT_HT2000_v5', 
    'HLT_HT2500_v5', 
    'HLT_L1_TripleJet_VBF_v5', 
    'HLT_PFHT125_v5', 
    'HLT_PFHT200_v6', 
    'HLT_PFHT250_v6', 
    'HLT_PFHT300_v7', 
    'HLT_PFHT350_v8', 
    'HLT_PFHT400_SixJet30_DoubleBTagCSV_p056_v6', 
    'HLT_PFHT400_SixJet30_v8', 
    'HLT_PFHT400_v7', 
    'HLT_PFHT450_SixJet40_BTagCSV_p056_v6', 
    'HLT_PFHT450_SixJet40_v8', 
    'HLT_PFHT475_v7', 
    'HLT_PFHT550_4JetPt50_v6', 
    'HLT_PFHT600_v8', 
    'HLT_PFHT650_4JetPt50_v6', 
    'HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v8', 
    'HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v8', 
    'HLT_PFHT650_v8', 
    'HLT_PFHT750_4JetPt70_v2', 
    'HLT_PFHT750_4JetPt80_v2', 
    'HLT_PFHT800_4JetPt50_v2', 
    'HLT_PFHT850_4JetPt50_v2', 
    'HLT_PFHT900_v6', 
    'HLT_PFJet140_v8', 
    'HLT_PFJet200_v8', 
    'HLT_PFJet260_v9', 
    'HLT_PFJet320_v9', 
    'HLT_PFJet400_v9', 
    'HLT_PFJet40_v9', 
    'HLT_PFJet450_v9', 
    'HLT_PFJet500_v9', 
    'HLT_PFJet60_v9', 
    'HLT_PFJet80_v8', 
    'HLT_QuadPFJet_VBF_v8', 
    'HLT_SingleCentralPFJet170_CFMax0p1_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetMET_selector
streamPhysicsHadronsTaus_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetMET_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetMET_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetMET_selector.triggerConditions = cms.vstring('HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDTight_BTagCSV_p067_v7', 
    'HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDTight_v8', 
    'HLT_DoubleMu3_PFMET50_v6', 
    'HLT_MET200_v5', 
    'HLT_MET250_v5', 
    'HLT_MET300_v5', 
    'HLT_MET600_v5', 
    'HLT_MET60_IsoTrk35_Loose_v3', 
    'HLT_MET700_v5', 
    'HLT_MET75_IsoTrk50_v6', 
    'HLT_MET90_IsoTrk50_v6', 
    'HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v8', 
    'HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v8', 
    'HLT_Mu6_PFHT200_PFMET100_v5', 
    'HLT_PFMET110_PFMHT110_IDTight_v8', 
    'HLT_PFMET120_PFMHT120_IDTight_v8', 
    'HLT_PFMET170_BeamHaloCleaned_v7', 
    'HLT_PFMET170_HBHECleaned_v9', 
    'HLT_PFMET170_HBHE_BeamHaloCleaned_v5', 
    'HLT_PFMET170_NotCleaned_v8', 
    'HLT_PFMET300_v7', 
    'HLT_PFMET400_v7', 
    'HLT_PFMET500_v7', 
    'HLT_PFMET600_v7', 
    'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v8', 
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v8', 
    'HLT_PFMETTypeOne190_HBHE_BeamHaloCleaned_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetTau_selector
streamPhysicsHadronsTaus_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetTau_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetTau_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg_v3', 
    'HLT_DoubleMediumCombinedIsoPFTau40_Trk1_eta2p1_Reg_v2', 
    'HLT_DoubleMediumCombinedIsoPFTau40_Trk1_eta2p1_v2', 
    'HLT_DoubleTightCombinedIsoPFTau35_Trk1_eta2p1_Reg_v3', 
    'HLT_DoubleTightCombinedIsoPFTau40_Trk1_eta2p1_Reg_v2', 
    'HLT_DoubleTightCombinedIsoPFTau40_Trk1_eta2p1_v2', 
    'HLT_IsoMu19_eta2p1_LooseCombinedIsoPFTau20_v1', 
    'HLT_IsoMu19_eta2p1_MediumCombinedIsoPFTau32_Trk1_eta2p1_Reg_v1', 
    'HLT_IsoMu19_eta2p1_TightCombinedIsoPFTau32_Trk1_eta2p1_Reg_v1', 
    'HLT_IsoMu21_eta2p1_MediumCombinedIsoPFTau32_Trk1_eta2p1_Reg_v1', 
    'HLT_IsoMu21_eta2p1_TightCombinedIsoPFTau32_Trk1_eta2p1_Reg_v1', 
    'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET110_v6', 
    'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120_v6', 
    'HLT_LooseIsoPFTau50_Trk30_eta2p1_MET90_v6', 
    'HLT_LooseIsoPFTau50_Trk30_eta2p1_v7', 
    'HLT_PFTau120_eta2p1_v5', 
    'HLT_PFTau140_eta2p1_v5', 
    'HLT_VLooseIsoPFTau120_Trk50_eta2p1_v5', 
    'HLT_VLooseIsoPFTau140_Trk50_eta2p1_v5')


# stream PhysicsMinimumBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias0_datasetL1MinimumBias0_selector
streamPhysicsMinimumBias0_datasetL1MinimumBias0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias0_datasetL1MinimumBias0_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias0_datasetL1MinimumBias0_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias0_datasetL1MinimumBias0_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part0_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias0_datasetL1MinimumBias1_selector
streamPhysicsMinimumBias0_datasetL1MinimumBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias0_datasetL1MinimumBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias0_datasetL1MinimumBias1_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias0_datasetL1MinimumBias1_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part1_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias0_datasetL1MinimumBias2_selector
streamPhysicsMinimumBias0_datasetL1MinimumBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias0_datasetL1MinimumBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias0_datasetL1MinimumBias2_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias0_datasetL1MinimumBias2_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part2_v2')


# stream PhysicsMinimumBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias1_datasetL1MinimumBias3_selector
streamPhysicsMinimumBias1_datasetL1MinimumBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias1_datasetL1MinimumBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias1_datasetL1MinimumBias3_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias1_datasetL1MinimumBias3_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part3_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias1_datasetL1MinimumBias4_selector
streamPhysicsMinimumBias1_datasetL1MinimumBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias1_datasetL1MinimumBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias1_datasetL1MinimumBias4_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias1_datasetL1MinimumBias4_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part4_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias1_datasetL1MinimumBias5_selector
streamPhysicsMinimumBias1_datasetL1MinimumBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias1_datasetL1MinimumBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias1_datasetL1MinimumBias5_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias1_datasetL1MinimumBias5_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part5_v2')


# stream PhysicsMinimumBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias2_datasetL1MinimumBias6_selector
streamPhysicsMinimumBias2_datasetL1MinimumBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias2_datasetL1MinimumBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias2_datasetL1MinimumBias6_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias2_datasetL1MinimumBias6_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part6_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias2_datasetL1MinimumBias7_selector
streamPhysicsMinimumBias2_datasetL1MinimumBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias2_datasetL1MinimumBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias2_datasetL1MinimumBias7_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias2_datasetL1MinimumBias7_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part7_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias2_datasetL1MinimumBias8_selector
streamPhysicsMinimumBias2_datasetL1MinimumBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias2_datasetL1MinimumBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias2_datasetL1MinimumBias8_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias2_datasetL1MinimumBias8_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part8_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMinimumBias2_datasetL1MinimumBias9_selector
streamPhysicsMinimumBias2_datasetL1MinimumBias9_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMinimumBias2_datasetL1MinimumBias9_selector.l1tResults = cms.InputTag('')
streamPhysicsMinimumBias2_datasetL1MinimumBias9_selector.throw      = cms.bool(False)
streamPhysicsMinimumBias2_datasetL1MinimumBias9_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF_OR_part9_v2')


# stream PhysicsMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetCharmonium_selector
streamPhysicsMuons_datasetCharmonium_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetCharmonium_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetCharmonium_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetCharmonium_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_Muon_v5', 
    'HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing_v4', 
    'HLT_Dimuon13_PsiPrime_v6', 
    'HLT_Dimuon16_Jpsi_v6', 
    'HLT_Dimuon20_Jpsi_v6', 
    'HLT_Dimuon6_Jpsi_NoVertexing_v4', 
    'HLT_Dimuon8_PsiPrime_Barrel_v6', 
    'HLT_DoubleMu4_3_Bs_v7', 
    'HLT_DoubleMu4_3_Jpsi_Displaced_v7', 
    'HLT_DoubleMu4_JpsiTrk_Displaced_v7', 
    'HLT_DoubleMu4_PsiPrimeTrk_Displaced_v7', 
    'HLT_Mu7p5_L2Mu2_Jpsi_v4', 
    'HLT_Mu7p5_Track2_Jpsi_v4', 
    'HLT_Mu7p5_Track3p5_Jpsi_v4', 
    'HLT_Mu7p5_Track7_Jpsi_v4', 
    'HLT_QuadMuon0_Dimuon0_Jpsi_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuon_selector
streamPhysicsMuons_datasetDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuon_selector.triggerConditions = cms.vstring('HLT_DoubleMu0_v2', 
    'HLT_DoubleMu18NoFiltersNoVtx_v5', 
    'HLT_DoubleMu23NoFiltersNoVtxDisplaced_v5', 
    'HLT_DoubleMu28NoFiltersNoVtxDisplaced_v5', 
    'HLT_DoubleMu33NoFiltersNoVtx_v5', 
    'HLT_DoubleMu38NoFiltersNoVtx_v5', 
    'HLT_DoubleMu8_Mass8_PFHT300_v9', 
    'HLT_L2DoubleMu23_NoVertex_v6', 
    'HLT_L2DoubleMu28_NoVertex_2Cha_Angle2p5_Mass10_v6', 
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_Mass10_v6', 
    'HLT_Mu10_CentralPFJet30_BTagCSV_p13_v5', 
    'HLT_Mu17_Mu8_DZ_v7', 
    'HLT_Mu17_Mu8_SameSign_DZ_v6', 
    'HLT_Mu17_Mu8_SameSign_v5', 
    'HLT_Mu17_Mu8_v5', 
    'HLT_Mu17_TkMu8_DZ_v6', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v7', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v6', 
    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v6', 
    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v5', 
    'HLT_Mu17_TrkIsoVVL_v4', 
    'HLT_Mu17_v4', 
    'HLT_Mu20_Mu10_DZ_v6', 
    'HLT_Mu20_Mu10_SameSign_DZ_v6', 
    'HLT_Mu20_Mu10_SameSign_v4', 
    'HLT_Mu20_Mu10_v5', 
    'HLT_Mu27_TkMu8_v5', 
    'HLT_Mu30_TkMu11_v5', 
    'HLT_Mu3_PFJet40_v6', 
    'HLT_Mu40_TkMu11_v5', 
    'HLT_Mu8_TrkIsoVVL_v5', 
    'HLT_Mu8_v5', 
    'HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v3', 
    'HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v3', 
    'HLT_TripleMu_12_10_5_v4', 
    'HLT_TripleMu_5_3_3_DZ_Mass3p8_v1', 
    'HLT_TrkMu15_DoubleTrkMu5NoFiltersNoVtx_v6', 
    'HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuonLowMass_selector
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuonLowMass_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_Trk_Tau3mu_v4', 
    'HLT_DoubleMu4_LowMassNonResonantTrk_Displaced_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetMuOnia_selector
streamPhysicsMuons_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetMuOnia_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Phi_Barrel_v6', 
    'HLT_Dimuon0_Upsilon_Muon_v5', 
    'HLT_Dimuon13_Upsilon_v6', 
    'HLT_Dimuon8_Upsilon_Barrel_v6', 
    'HLT_Mu25_TkMu0_dEta18_Onia_v6', 
    'HLT_Mu7p5_L2Mu2_Upsilon_v4', 
    'HLT_Mu7p5_Track2_Upsilon_v4', 
    'HLT_Mu7p5_Track3p5_Upsilon_v4', 
    'HLT_Mu7p5_Track7_Upsilon_v4', 
    'HLT_QuadMuon0_Dimuon0_Upsilon_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetMuonEG_selector
streamPhysicsMuons_datasetMuonEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetMuonEG_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetMuonEG_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetMuonEG_selector.triggerConditions = cms.vstring('HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v8', 
    'HLT_Mu12_Photon25_CaloIdL_L1ISO_v8', 
    'HLT_Mu12_Photon25_CaloIdL_L1OR_v8', 
    'HLT_Mu12_Photon25_CaloIdL_v8', 
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v4', 
    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v3', 
    'HLT_Mu17_Photon30_CaloIdL_L1ISO_v9', 
    'HLT_Mu17_Photon35_CaloIdL_L1ISO_v9', 
    'HLT_Mu17_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v9', 
    'HLT_Mu23NoFiltersNoVtx_Photon23_CaloIdL_v7', 
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v4', 
    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v9', 
    'HLT_Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ_v3', 
    'HLT_Mu27_Ele37_CaloIdL_GsfTrkIdVL_v6', 
    'HLT_Mu28NoFiltersNoVtxDisplaced_Photon28_CaloIdL_v7', 
    'HLT_Mu33NoFiltersNoVtxDisplaced_Photon33_CaloIdL_v7', 
    'HLT_Mu33_Ele33_CaloIdL_GsfTrkIdVL_v3', 
    'HLT_Mu37_Ele27_CaloIdL_GsfTrkIdVL_v6', 
    'HLT_Mu38NoFiltersNoVtx_Photon38_CaloIdL_v7', 
    'HLT_Mu42NoFiltersNoVtx_Photon42_CaloIdL_v7', 
    'HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v9', 
    'HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT300_v10', 
    'HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v9', 
    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetSingleMuon_selector
streamPhysicsMuons_datasetSingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetSingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetSingleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetSingleMuon_selector.triggerConditions = cms.vstring('HLT_DoubleIsoMu17_eta2p1_noDzCut_v5', 
    'HLT_IsoMu16_eta2p1_MET30_LooseIsoPFTau50_Trk30_eta2p1_v5', 
    'HLT_IsoMu16_eta2p1_MET30_v4', 
    'HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1_v5', 
    'HLT_IsoMu19_eta2p1_LooseIsoPFTau20_v5', 
    'HLT_IsoMu19_eta2p1_MediumIsoPFTau32_Trk1_eta2p1_Reg_v5', 
    'HLT_IsoMu20_v6', 
    'HLT_IsoMu21_eta2p1_LooseIsoPFTau20_SingleL1_v5', 
    'HLT_IsoMu21_eta2p1_LooseIsoPFTau50_Trk30_eta2p1_SingleL1_v4', 
    'HLT_IsoMu21_eta2p1_MediumIsoPFTau32_Trk1_eta2p1_Reg_v5', 
    'HLT_IsoMu22_eta2p1_v4', 
    'HLT_IsoMu22_v5', 
    'HLT_IsoMu24_eta2p1_v3', 
    'HLT_IsoMu24_v4', 
    'HLT_IsoMu27_v7', 
    'HLT_IsoTkMu20_v7', 
    'HLT_IsoTkMu22_eta2p1_v4', 
    'HLT_IsoTkMu22_v5', 
    'HLT_IsoTkMu24_eta2p1_v3', 
    'HLT_IsoTkMu24_v4', 
    'HLT_IsoTkMu27_v7', 
    'HLT_L1SingleMu18_v1', 
    'HLT_L2Mu10_v3', 
    'HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v5', 
    'HLT_Mu15_IsoVVVL_BTagCSV_p067_PFHT400_v6', 
    'HLT_Mu15_IsoVVVL_PFHT400_PFMET50_v5', 
    'HLT_Mu15_IsoVVVL_PFHT400_v5', 
    'HLT_Mu15_IsoVVVL_PFHT600_v8', 
    'HLT_Mu20_v4', 
    'HLT_Mu24_eta2p1_v5', 
    'HLT_Mu27_v5', 
    'HLT_Mu28NoFiltersNoVtx_CentralCaloJet40_v5', 
    'HLT_Mu28NoFiltersNoVtx_DisplacedJet40_Loose_v5', 
    'HLT_Mu300_v3', 
    'HLT_Mu30_eta2p1_PFJet150_PFJet50_v5', 
    'HLT_Mu350_v3', 
    'HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Loose_v5', 
    'HLT_Mu38NoFiltersNoVtxDisplaced_DisplacedJet60_Tight_v5', 
    'HLT_Mu38NoFiltersNoVtx_DisplacedJet60_Loose_v5', 
    'HLT_Mu40_eta2p1_PFJet200_PFJet50_v7', 
    'HLT_Mu45_eta2p1_v5', 
    'HLT_Mu50_IsoVVVL_PFHT400_v5', 
    'HLT_Mu50_v5', 
    'HLT_Mu55_v4', 
    'HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v4', 
    'HLT_TkMu17_v1', 
    'HLT_TkMu20_v4', 
    'HLT_TkMu24_eta2p1_v5', 
    'HLT_TkMu27_v5', 
    'HLT_TkMu50_v3')


# stream PhysicsParkingScoutingMonitor

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsParkingScoutingMonitor_datasetParkingScoutingMonitor_selector
streamPhysicsParkingScoutingMonitor_datasetParkingScoutingMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsParkingScoutingMonitor_datasetParkingScoutingMonitor_selector.l1tResults = cms.InputTag('')
streamPhysicsParkingScoutingMonitor_datasetParkingScoutingMonitor_selector.throw      = cms.bool(False)
streamPhysicsParkingScoutingMonitor_datasetParkingScoutingMonitor_selector.triggerConditions = cms.vstring('DST_CaloJet40_BTagScouting_v7', 
    'DST_CaloJet40_CaloBTagScouting_v6', 
    'DST_CaloJet40_CaloScouting_PFScouting_v7', 
    'DST_DoubleMu3_Mass10_BTagScouting_v8', 
    'DST_DoubleMu3_Mass10_CaloScouting_PFScouting_v7', 
    'DST_HT250_CaloBTagScouting_v3', 
    'DST_HT250_CaloScouting_v5', 
    'DST_HT410_BTagScouting_v7', 
    'DST_HT410_PFScouting_v7', 
    'DST_HT450_BTagScouting_v7', 
    'DST_HT450_PFScouting_v7', 
    'DST_L1DoubleMu_BTagScouting_v8', 
    'DST_L1DoubleMu_CaloScouting_PFScouting_v7', 
    'DST_L1HTT_BTagScouting_v7', 
    'DST_L1HTT_CaloBTagScouting_v6', 
    'DST_L1HTT_CaloScouting_PFScouting_v7', 
    'DST_ZeroBias_BTagScouting_v7', 
    'DST_ZeroBias_CaloScouting_PFScouting_v6', 
    'HLT_HT430to450_v4', 
    'HLT_HT450to470_v4', 
    'HLT_HT470to500_v4', 
    'HLT_HT500to550_v4', 
    'HLT_HT550to650_v4', 
    'HLT_HT650_v5')

