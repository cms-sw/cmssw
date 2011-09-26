# /dev/CMSSW_4_2_0/GRun/V274

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTag_selector
streamA_datasetBTag_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTag_selector.l1tResults = cms.InputTag('')
streamA_datasetBTag_selector.throw      = cms.bool(False)
streamA_datasetBTag_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet110_Mu5_v12', 
    'HLT_BTagMu_DiJet20_Mu5_v12', 
    'HLT_BTagMu_DiJet40_Mu5_v12', 
    'HLT_BTagMu_DiJet70_Mu5_v12', 
    'HLT_R014_MR200_CentralJet40_BTagIP_v2', 
    'HLT_R014_MR400_CentralJet40_BTagIP_v2', 
    'HLT_R014_MR450_CentralJet40_BTagIP_v2', 
    'HLT_R020_MR300_CentralJet40_BTagIP_v2', 
    'HLT_R020_MR350_CentralJet40_BTagIP_v2', 
    'HLT_R030_MR200_CentralJet40_BTagIP_v2', 
    'HLT_R030_MR250_CentralJet40_BTagIP_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v8', 
    'HLT_BeamGas_HF_Beam1_v2', 
    'HLT_BeamGas_HF_Beam2_v2', 
    'HLT_BeamGas_HF_v6', 
    'HLT_IsoTrackHB_v8', 
    'HLT_IsoTrackHE_v9', 
    'HLT_L1SingleEG12_v3', 
    'HLT_L1SingleEG5_v3', 
    'HLT_L1SingleJet16_v4', 
    'HLT_L1SingleJet36_v4', 
    'HLT_L1SingleMuOpen_DT_v4', 
    'HLT_L1SingleMuOpen_v4', 
    'HLT_L1_Interbunch_BSC_v3', 
    'HLT_L1_PreCollisions_v3', 
    'HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v12')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v7', 
    'HLT_L1SingleMuOpen_AntiBPTX_v3', 
    'HLT_L1TrackerCosmics_v4', 
    'HLT_RegionalCosmicTracking_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v4', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v9', 
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass30_v8', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v9', 
    'HLT_Ele22_CaloIdL_CaloIsoVL_Ele15_HFT_v2', 
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_Ele17_v2', 
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_v7', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v10', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v9', 
    'HLT_Ele8_CaloIdL_TrkIdVL_v9', 
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v7', 
    'HLT_Ele8_v9', 
    'HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v10', 
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_v13', 
    'HLT_DoubleMu45_v9', 
    'HLT_DoubleMu5_IsoMu5_v11', 
    'HLT_DoubleMu5_v4', 
    'HLT_DoubleMu6_v11', 
    'HLT_DoubleMu7_Acoplanarity03_v3', 
    'HLT_DoubleMu7_v11', 
    'HLT_L1DoubleMu0_v4', 
    'HLT_L2DoubleMu0_v7', 
    'HLT_L2DoubleMu23_NoVertex_v8', 
    'HLT_L2DoubleMu30_NoVertex_v4', 
    'HLT_L2DoubleMu45_NoVertex_v1', 
    'HLT_Mu13_Mu8_v10', 
    'HLT_Mu17_Mu8_v10', 
    'HLT_Mu17_TkMu8_v3', 
    'HLT_Mu8_Jet40_v13', 
    'HLT_TripleMu5_v12')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_DoubleEle8_CaloIdT_TrkIdVL_HT150_v8', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_HT150_v2', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_HT200_v2', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_v4', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R014_MR200_v2', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R025_MR200_v3', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R029_MR200_v3', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R033_MR200_v2', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT40_v4', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT50_v3', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_v3', 
    'HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta3_Jet20_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_BTagIP_v7', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_v7', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_v1', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralPFJet30_v1', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_QuadCentralPFJet30_v1', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFJet30_v1', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_BTagIP_v11', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralPFJet30_v1', 
    'HLT_Ele25_CaloIdVT_TrkIdT_DiCentralPFJet30_v1', 
    'HLT_Ele25_CaloIdVT_TrkIdT_QuadCentralPFJet30_v1', 
    'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFJet30_v1', 
    'HLT_Ele27_CaloIdVT_TrkIdT_DiCentralPFJet25_v3', 
    'HLT_Ele27_CaloIdVT_TrkIdT_DiPFJet25_Deta3_v3', 
    'HLT_Ele27_WP80_DiCentralPFJet25_PFMHT15_v3', 
    'HLT_Ele27_WP80_DiCentralPFJet25_v3', 
    'HLT_Ele27_WP80_DiPFJet25_Deta3_v3', 
    'HLT_Ele32_WP80_DiCentralPFJet25_PFMHT25_v3', 
    'HLT_Ele32_WP80_DiPFJet25_Deta3p5_v3', 
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v7', 
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v7', 
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v7', 
    'HLT_HT350_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT45_v9', 
    'HLT_HT400_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT50_v3', 
    'HLT_HT400_Ele60_CaloIdT_TrkIdT_v3', 
    'HLT_HT450_Ele60_CaloIdT_TrkIdT_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFEDMonitor_selector
streamA_datasetFEDMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFEDMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetFEDMonitor_selector.throw      = cms.bool(False)
streamA_datasetFEDMonitor_selector.triggerConditions = cms.vstring('HLT_DTErrors_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHT_selector
streamA_datasetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHT_selector.throw      = cms.bool(False)
streamA_datasetHT_selector.triggerConditions = cms.vstring('HLT_DiJet130_PT130_v8', 
    'HLT_DiJet160_PT160_v8', 
    'HLT_FatJetMass750_DR1p1_Deta2p0_v4', 
    'HLT_FatJetMass850_DR1p1_Deta2p0_v4', 
    'HLT_HT150_L1FastJet_v2', 
    'HLT_HT150_v10', 
    'HLT_HT2000_v4', 
    'HLT_HT200_L1FastJet_v2', 
    'HLT_HT200_v10', 
    'HLT_HT250_AlphaT0p58_v2', 
    'HLT_HT250_AlphaT0p60_v2', 
    'HLT_HT250_DoubleDisplacedJet60_PromptTrack_v8', 
    'HLT_HT250_DoubleDisplacedJet60_v10', 
    'HLT_HT250_L1FastJet_v2', 
    'HLT_HT250_MHT100_L1FastJet_v2', 
    'HLT_HT250_MHT100_v4', 
    'HLT_HT250_MHT90_L1FastJet_v2', 
    'HLT_HT250_MHT90_v4', 
    'HLT_HT250_v10', 
    'HLT_HT300_AlphaT0p54_v4', 
    'HLT_HT300_AlphaT0p55_v2', 
    'HLT_HT300_CentralJet30_BTagIP_PFMHT55_v11', 
    'HLT_HT300_CentralJet30_BTagIP_PFMHT65_v4', 
    'HLT_HT300_CentralJet30_BTagIP_v9', 
    'HLT_HT300_L1FastJet_v2', 
    'HLT_HT300_MHT80_L1FastJet_v2', 
    'HLT_HT300_MHT80_v4', 
    'HLT_HT300_MHT90_L1FastJet_v2', 
    'HLT_HT300_MHT90_v4', 
    'HLT_HT300_PFMHT55_v11', 
    'HLT_HT300_PFMHT65_v4', 
    'HLT_HT300_v11', 
    'HLT_HT350_L1FastJet_v2', 
    'HLT_HT350_MHT100_L1FastJet_v2', 
    'HLT_HT350_MHT100_v2', 
    'HLT_HT350_MHT110_L1FastJet_v2', 
    'HLT_HT350_MHT110_v2', 
    'HLT_HT350_MHT70_L1FastJet_v2', 
    'HLT_HT350_MHT70_v4', 
    'HLT_HT350_MHT80_L1FastJet_v2', 
    'HLT_HT350_MHT80_v4', 
    'HLT_HT350_MHT90_L1FastJet_v2', 
    'HLT_HT350_MHT90_v3', 
    'HLT_HT350_v10', 
    'HLT_HT400_AlphaT0p51_v9', 
    'HLT_HT400_AlphaT0p52_v4', 
    'HLT_HT400_L1FastJet_v2', 
    'HLT_HT400_MHT100_L1FastJet_v2', 
    'HLT_HT400_MHT100_v2', 
    'HLT_HT400_MHT80_L1FastJet_v2', 
    'HLT_HT400_MHT80_v3', 
    'HLT_HT400_MHT90_L1FastJet_v2', 
    'HLT_HT400_MHT90_v2', 
    'HLT_HT400_v10', 
    'HLT_HT450_AlphaT0p51_v4', 
    'HLT_HT450_L1FastJet_v2', 
    'HLT_HT450_v10', 
    'HLT_HT500_L1FastJet_v2', 
    'HLT_HT500_v10', 
    'HLT_HT550_L1FastJet_v2', 
    'HLT_HT550_v10', 
    'HLT_HT600_L1FastJet_v2', 
    'HLT_HT600_v3', 
    'HLT_HT650_L1FastJet_v2', 
    'HLT_HT650_v3', 
    'HLT_HT750_L1FastJet_v2', 
    'HLT_HT750_v2', 
    'HLT_R020_MR550_v9', 
    'HLT_R025_MR450_v9', 
    'HLT_R033_MR350_v9', 
    'HLT_R038_MR250_v9', 
    'HLT_RMR65_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v5', 
    'HLT_L1Tech_HBHEHO_totalOR_v3', 
    'HLT_L1Tech_HCAL_HF_single_channel_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v7', 
    'HLT_HcalPhiSym_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighPileUp_selector
streamA_datasetHighPileUp_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighPileUp_selector.l1tResults = cms.InputTag('')
streamA_datasetHighPileUp_selector.throw      = cms.bool(False)
streamA_datasetHighPileUp_selector.triggerConditions = cms.vstring('HLT_300Tower0p5_v1', 
    'HLT_300Tower0p6_v1', 
    'HLT_300Tower0p7_v1', 
    'HLT_300Tower0p8_v1', 
    'HLT_70Jet10_v3', 
    'HLT_70Jet13_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJet_selector
streamA_datasetJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJet_selector.l1tResults = cms.InputTag('')
streamA_datasetJet_selector.throw      = cms.bool(False)
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiJetAve110_v8', 
    'HLT_DiJetAve190_v8', 
    'HLT_DiJetAve240_v8', 
    'HLT_DiJetAve300_v9', 
    'HLT_DiJetAve30_v8', 
    'HLT_DiJetAve370_v9', 
    'HLT_DiJetAve60_v8', 
    'HLT_Jet110_L1FastJet_v2', 
    'HLT_Jet110_v8', 
    'HLT_Jet190_L1FastJet_v2', 
    'HLT_Jet190_v8', 
    'HLT_Jet240_CentralJet30_BTagIP_v5', 
    'HLT_Jet240_L1FastJet_v2', 
    'HLT_Jet240_v8', 
    'HLT_Jet270_CentralJet30_BTagIP_v5', 
    'HLT_Jet300_L1FastJet_v2', 
    'HLT_Jet300_v8', 
    'HLT_Jet30_L1FastJet_v2', 
    'HLT_Jet30_v8', 
    'HLT_Jet370_L1FastJet_v2', 
    'HLT_Jet370_NoJetID_v9', 
    'HLT_Jet370_v9', 
    'HLT_Jet60_L1FastJet_v2', 
    'HLT_Jet60_v8', 
    'HLT_Jet800_L1FastJet_v2', 
    'HLT_Jet800_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetLogMonitor_selector
streamA_datasetLogMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetLogMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetLogMonitor_selector.throw      = cms.bool(False)
streamA_datasetLogMonitor_selector.triggerConditions = cms.vstring('HLT_LogMonitor_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMET_selector
streamA_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetMET_selector.throw      = cms.bool(False)
streamA_datasetMET_selector.triggerConditions = cms.vstring('HLT_CentralJet80_MET110_v2', 
    'HLT_CentralJet80_MET65_v9', 
    'HLT_CentralJet80_MET80_v8', 
    'HLT_CentralJet80_MET95_v2', 
    'HLT_DiCentralJet20_BTagIP_MET65_v9', 
    'HLT_DiCentralJet20_MET100_HBHENoiseFiltered_v3', 
    'HLT_DiCentralJet20_MET80_v7', 
    'HLT_DiJet60_MET45_v9', 
    'HLT_IsoMu15_L1ETM20_v3', 
    'HLT_L1ETM30_v4', 
    'HLT_L2Mu60_1Hit_MET40_v5', 
    'HLT_L2Mu60_1Hit_MET60_v5', 
    'HLT_MET100_HBHENoiseFiltered_v6', 
    'HLT_MET100_v7', 
    'HLT_MET120_HBHENoiseFiltered_v6', 
    'HLT_MET120_v7', 
    'HLT_MET200_HBHENoiseFiltered_v6', 
    'HLT_MET200_v7', 
    'HLT_MET400_v2', 
    'HLT_MET65_HBHENoiseFiltered_v5', 
    'HLT_MET65_v4', 
    'HLT_Mu15_L1ETM20_v3', 
    'HLT_PFMHT150_v15')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v9', 
    'HLT_JetE30_NoBPTX_NoHalo_v9', 
    'HLT_JetE30_NoBPTX_v7', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v4', 
    'HLT_Physics_v2', 
    'HLT_PixelTracks_Multiplicity100_v8', 
    'HLT_PixelTracks_Multiplicity80_v8', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_Ele8_CaloIdT_TrkIdT_v3', 
    'HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v7', 
    'HLT_Mu15_DoublePhoton15_CaloIdL_v13', 
    'HLT_Mu15_Photon20_CaloIdL_v13', 
    'HLT_Mu17_Ele8_CaloIdL_v12', 
    'HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_v7', 
    'HLT_Mu5_DoubleEle8_CaloIdT_TrkIdVL_v7', 
    'HLT_Mu5_Ele8_CaloIdT_CaloIsoVL_v4', 
    'HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v7', 
    'HLT_Mu8_Ele17_CaloIdL_v12', 
    'HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_v7', 
    'HLT_Mu8_Photon20_CaloIdVT_IsoT_v12')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_Mass8_HT150_v3', 
    'HLT_DoubleMu8_Mass8_HT150_v3', 
    'HLT_DoubleMu8_Mass8_HT200_v3', 
    'HLT_DoubleTkIso10Mu5_Mass8_HT150_v3', 
    'HLT_DoubleTkIso10Mu5_Mass8_HT200_v3', 
    'HLT_HT300_Mu15_PFMHT40_v4', 
    'HLT_HT300_Mu15_PFMHT50_v3', 
    'HLT_HT350_Mu5_PFMHT45_v11', 
    'HLT_HT400_Mu5_PFMHT50_v3', 
    'HLT_IsoMu17_eta2p1_CentralJet30_BTagIP_v4', 
    'HLT_IsoMu17_eta2p1_CentralJet30_v4', 
    'HLT_IsoMu17_eta2p1_CentralPFJet30_v1', 
    'HLT_IsoMu17_eta2p1_DiCentralPFJet25_PFMHT15_v3', 
    'HLT_IsoMu17_eta2p1_DiCentralPFJet25_PFMHT25_v3', 
    'HLT_IsoMu17_eta2p1_DiCentralPFJet25_v3', 
    'HLT_IsoMu17_eta2p1_DiCentralPFJet30_v1', 
    'HLT_IsoMu17_eta2p1_DiPFJet25_Deta3_PFJet25_v3', 
    'HLT_IsoMu17_eta2p1_DiPFJet25_Deta3_v3', 
    'HLT_IsoMu17_eta2p1_QuadCentralPFJet30_v1', 
    'HLT_IsoMu17_eta2p1_TriCentralPFJet30_v1', 
    'HLT_IsoMu20_DiCentralJet34_v6', 
    'HLT_Mu10_R014_MR200_v3', 
    'HLT_Mu10_R025_MR200_v4', 
    'HLT_Mu10_R029_MR200_v4', 
    'HLT_Mu10_R033_MR200_v3', 
    'HLT_Mu12_eta2p1_DiCentralJet20_BTagIP3D1stTrack_v4', 
    'HLT_Mu12_eta2p1_DiCentralJet20_DiBTagIP3D1stTrack_v4', 
    'HLT_Mu12_eta2p1_DiCentralJet30_BTagIP3D_v4', 
    'HLT_Mu17_eta2p1_CentralJet30_BTagIP_v4', 
    'HLT_Mu17_eta2p1_CentralPFJet30_v1', 
    'HLT_Mu17_eta2p1_DiCentralPFJet25_PFMHT15_v3', 
    'HLT_Mu17_eta2p1_DiCentralPFJet30_v1', 
    'HLT_Mu17_eta2p1_DiPFJet25_Deta3_v3', 
    'HLT_Mu17_eta2p1_QuadCentralPFJet30_v1', 
    'HLT_Mu17_eta2p1_TriCentralPFJet30_v1', 
    'HLT_Mu40_HT300_v3', 
    'HLT_Mu5_DiJet30_v4', 
    'HLT_Mu5_Ele8_CaloIdT_TrkIdVL_HT150_v4', 
    'HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_HT150_v3', 
    'HLT_Mu5_QuadJet30_v4', 
    'HLT_Mu5_TriJet30_v4', 
    'HLT_Mu60_HT300_v3', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_HT150_v3', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_HT200_v3', 
    'HLT_TkIso10Mu5_Ele8_CaloIdT_CaloIsoVVL_TrkIdVL_Mass8_HT150_v3', 
    'HLT_TkIso10Mu5_Ele8_CaloIdT_CaloIsoVVL_TrkIdVL_Mass8_HT200_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_Muon_v10', 
    'HLT_Dimuon0_Jpsi_NoVertexing_v6', 
    'HLT_Dimuon0_Jpsi_v9', 
    'HLT_Dimuon0_Omega_Phi_v3', 
    'HLT_Dimuon0_Upsilon_Muon_v10', 
    'HLT_Dimuon0_Upsilon_v9', 
    'HLT_Dimuon10_Jpsi_Barrel_v9', 
    'HLT_Dimuon11_PsiPrime_v4', 
    'HLT_Dimuon13_Jpsi_Barrel_v4', 
    'HLT_Dimuon6_LowMass_v4', 
    'HLT_Dimuon7_Upsilon_Barrel_v4', 
    'HLT_Dimuon9_PsiPrime_v4', 
    'HLT_Dimuon9_Upsilon_Barrel_v4', 
    'HLT_DoubleMu4_Dimuon4_Bs_Barrel_v4', 
    'HLT_DoubleMu4_Dimuon6_Bs_v4', 
    'HLT_DoubleMu4_Jpsi_Displaced_v4', 
    'HLT_DoubleMu4p5_LowMass_Displaced_v4', 
    'HLT_DoubleMu5_Jpsi_Displaced_v4', 
    'HLT_DoubleMu5_LowMass_Displaced_v4', 
    'HLT_Mu5_L2Mu2_Jpsi_v12', 
    'HLT_Mu5_Track2_Jpsi_v12', 
    'HLT_Mu7_Track7_Jpsi_v13')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_CentralJet46_CentralJet38_CentralJet20_DiBTagIP3D_v3', 
    'HLT_CentralJet46_CentralJet38_DiBTagIP3D_v5', 
    'HLT_CentralJet60_CentralJet53_DiBTagIP3D_v4', 
    'HLT_DiCentralJet36_BTagIP3DLoose_v3', 
    'HLT_DoubleJet30_ForwardBackward_v9', 
    'HLT_DoubleJet60_ForwardBackward_v9', 
    'HLT_DoubleJet70_ForwardBackward_v9', 
    'HLT_DoubleJet80_ForwardBackward_v9', 
    'HLT_EightJet120_v4', 
    'HLT_EightJet35_v2', 
    'HLT_EightJet40_v2', 
    'HLT_ExclDiJet60_HFAND_v8', 
    'HLT_ExclDiJet60_HFOR_v8', 
    'HLT_L1DoubleJet36Central_v4', 
    'HLT_L1MultiJet_v4', 
    'HLT_QuadJet40_IsoPFTau40_v16', 
    'HLT_QuadJet40_v10', 
    'HLT_QuadJet45_DiJet40_v2', 
    'HLT_QuadJet45_IsoPFTau45_v11', 
    'HLT_QuadJet50_DiJet40_v4', 
    'HLT_QuadJet50_IsoPFTau50_v5', 
    'HLT_QuadJet50_Jet40_Jet30_v6', 
    'HLT_QuadJet70_v9', 
    'HLT_QuadJet80_v4', 
    'HLT_QuadJet90_v2', 
    'HLT_SixJet45_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoubleEle33_CaloIdL_CaloIsoT_v2', 
    'HLT_DoubleEle33_CaloIdL_v6', 
    'HLT_DoubleEle33_CaloIdT_v2', 
    'HLT_DoubleEle45_CaloIdL_v5', 
    'HLT_DoublePhoton38_HEVT_v3', 
    'HLT_DoublePhoton40_CaloIdL_MR150_v2', 
    'HLT_DoublePhoton40_CaloIdL_R014_MR150_v2', 
    'HLT_DoublePhoton43_HEVT_v1', 
    'HLT_DoublePhoton48_HEVT_v1', 
    'HLT_DoublePhoton5_IsoVL_CEP_v8', 
    'HLT_DoublePhoton60_v4', 
    'HLT_DoublePhoton70_v1', 
    'HLT_DoublePhoton80_v2', 
    'HLT_Photon135_v2', 
    'HLT_Photon200_NoHE_v4', 
    'HLT_Photon20_CaloIdVL_IsoL_v8', 
    'HLT_Photon20_R9Id_Photon18_R9Id_v7', 
    'HLT_Photon225_NoHE_v2', 
    'HLT_Photon26_CaloIdXL_IsoXL_Photon18_CaloIdXL_IsoXL_Mass60_v2', 
    'HLT_Photon26_CaloIdXL_IsoXL_Photon18_R9IdT_Mass60_v2', 
    'HLT_Photon26_CaloIdXL_IsoXL_Photon18_v2', 
    'HLT_Photon26_Photon18_v7', 
    'HLT_Photon26_R9IdT_Photon18_CaloIdXL_IsoXL_Mass60_v2', 
    'HLT_Photon26_R9IdT_Photon18_R9IdT_Mass60_v1', 
    'HLT_Photon30_CaloIdVL_IsoL_v10', 
    'HLT_Photon30_CaloIdVL_v8', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_CaloIdL_IsoVL_v5', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_R9Id_v4', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_v6', 
    'HLT_Photon36_CaloIdVL_Photon22_CaloIdVL_v2', 
    'HLT_Photon36_Photon22_v1', 
    'HLT_Photon36_R9Id_Photon22_CaloIdL_IsoVL_v5', 
    'HLT_Photon36_R9Id_Photon22_R9Id_v3', 
    'HLT_Photon400_v2', 
    'HLT_Photon50_CaloIdVL_IsoL_v8', 
    'HLT_Photon50_CaloIdVL_v4', 
    'HLT_Photon75_CaloIdVL_IsoL_v9', 
    'HLT_Photon75_CaloIdVL_v7', 
    'HLT_Photon90_CaloIdVL_IsoL_v6', 
    'HLT_Photon90_CaloIdVL_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon30_CaloIdVT_CentralJet20_BTagIP_v5', 
    'HLT_Photon40_CaloIdL_R017_MR500_v5', 
    'HLT_Photon40_CaloIdL_R023_MR350_v5', 
    'HLT_Photon40_CaloIdL_R029_MR250_v5', 
    'HLT_Photon40_CaloIdL_R042_MR200_v5', 
    'HLT_Photon55_CaloIdL_R017_MR500_v3', 
    'HLT_Photon55_CaloIdL_R023_MR350_v3', 
    'HLT_Photon55_CaloIdL_R029_MR250_v3', 
    'HLT_Photon55_CaloIdL_R042_MR200_v3', 
    'HLT_Photon60_CaloIdL_HT300_v2', 
    'HLT_Photon60_CaloIdL_MHT70_v2', 
    'HLT_Photon70_CaloIdXL_HT400_v2', 
    'HLT_Photon70_CaloIdXL_HT500_v2', 
    'HLT_Photon70_CaloIdXL_MHT100_v2', 
    'HLT_Photon70_CaloIdXL_MHT90_v2', 
    'HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet25_v3', 
    'HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet30_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele100_CaloIdVT_TrkIdT_v2', 
    'HLT_Ele25_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v6', 
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v2', 
    'HLT_Ele27_WP80_PFMT50_v7', 
    'HLT_Ele27_WP80_v2', 
    'HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v2', 
    'HLT_Ele32_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_v6', 
    'HLT_Ele32_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v8', 
    'HLT_Ele32_WP70_PFMT50_v7', 
    'HLT_Ele32_WP70_v2', 
    'HLT_Ele65_CaloIdVT_TrkIdT_v5', 
    'HLT_Ele80_CaloIdVT_TrkIdT_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_IsoMu15_eta2p1_v4', 
    'HLT_IsoMu15_v17', 
    'HLT_IsoMu17_v17', 
    'HLT_IsoMu20_v12', 
    'HLT_IsoMu24_eta2p1_v6', 
    'HLT_IsoMu24_v12', 
    'HLT_IsoMu30_eta2p1_v6', 
    'HLT_IsoMu34_eta2p1_v4', 
    'HLT_IsoMu40_eta2p1_v1', 
    'HLT_L1SingleMu10_v4', 
    'HLT_L1SingleMu20_v4', 
    'HLT_L2Mu10_v6', 
    'HLT_L2Mu20_v6', 
    'HLT_Mu100_eta2p1_v4', 
    'HLT_Mu12_v11', 
    'HLT_Mu15_v12', 
    'HLT_Mu200_eta2p1_v1', 
    'HLT_Mu20_v11', 
    'HLT_Mu24_eta2p1_v4', 
    'HLT_Mu24_v11', 
    'HLT_Mu30_eta2p1_v4', 
    'HLT_Mu30_v11', 
    'HLT_Mu40_eta2p1_v4', 
    'HLT_Mu40_v9', 
    'HLT_Mu50_eta2p1_v1', 
    'HLT_Mu5_v13', 
    'HLT_Mu60_eta2p1_v4', 
    'HLT_Mu8_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleIsoPFTau45_Trk5_eta2p1_v6', 
    'HLT_DoubleIsoPFTau55_Trk5_eta2p1_v3', 
    'HLT_IsoPFTau40_IsoPFTau30_Trk5_eta2p1_v6', 
    'HLT_MediumIsoPFTau35_Trk20_MET60_v4', 
    'HLT_MediumIsoPFTau35_Trk20_MET70_v4', 
    'HLT_MediumIsoPFTau35_Trk20_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele18_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v4', 
    'HLT_Ele18_CaloIdVT_TrkIdT_MediumIsoPFTau20_v4', 
    'HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau25_v3', 
    'HLT_HT350_DoubleIsoPFTau10_Trk3_PFMHT45_v11', 
    'HLT_HT400_DoubleIsoPFTau10_Trk3_PFMHT50_v3', 
    'HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v4', 
    'HLT_IsoMu15_eta2p1_MediumIsoPFTau20_v4', 
    'HLT_IsoMu15_eta2p1_TightIsoPFTau20_v4', 
    'HLT_Mu15_LooseIsoPFTau15_v12')

