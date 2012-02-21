# /dev/CMSSW_5_1_0/GRun/V62

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
    'HLT_L1SingleEG12_v4', 
    'HLT_L1SingleEG5_v4', 
    'HLT_L1SingleJet16_v5', 
    'HLT_L1SingleJet36_v5', 
    'HLT_L1SingleMu10_v5', 
    'HLT_L1SingleMu20_v5', 
    'HLT_L1SingleMuOpen_DT_v5', 
    'HLT_L1SingleMuOpen_v5', 
    'HLT_L1Tech_DT_GlobalOR_v2', 
    'HLT_L1_Interbunch_BSC_v4', 
    'HLT_L1_PreCollisions_v4', 
    'HLT_L2Mu10_v7', 
    'HLT_L2Mu20_v7', 
    'HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v14')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v9', 
    'HLT_L1SingleMuOpen_AntiBPTX_v4', 
    'HLT_L1TrackerCosmics_v5', 
    'HLT_L3MuonsCosmicTracking_v6', 
    'HLT_RegionalCosmicTracking_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v6', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v11', 
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v11', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass30_v10', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v11', 
    'HLT_Ele22_CaloIdL_CaloIsoVL_Ele15_HFT_v4', 
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_Ele17_v4', 
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_v9', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v13', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v11', 
    'HLT_Ele8_CaloIdL_TrkIdVL_v11', 
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9', 
    'HLT_Ele8_v11', 
    'HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v12', 
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v12')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu11_Acoplanarity03_v1', 
    'HLT_DoubleMu5_IsoMu5_v13', 
    'HLT_L1DoubleMu0_v5', 
    'HLT_L2DoubleMu0_v8', 
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v1', 
    'HLT_L2DoubleMu23_NoVertex_v9', 
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v1', 
    'HLT_Mu17_Mu8_v12', 
    'HLT_Mu17_TkMu8_v5', 
    'HLT_Mu22_TkMu22_v1', 
    'HLT_Mu22_TkMu8_v1', 
    'HLT_Mu8_Jet40_v16', 
    'HLT_TripleMu5_v14')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_HT150_v5', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_HT200_v5', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_v6', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R014_MR200_v5', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R025_MR200_v6', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R029_MR200_v6', 
    'HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R033_MR200_v5', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT40_v7', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT50_v6', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_v5', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_BTagIP_v10', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_v10', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_BTagIPIter_v1', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_QuadCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_TrkIdT_QuadCentralPFJet30_v4', 
    'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFJet30_v4', 
    'HLT_Ele27_CaloIdVT_TrkIdT_DiCentralPFJet25_v6', 
    'HLT_Ele27_CaloIdVT_TrkIdT_DiPFJet25_Deta3_v6', 
    'HLT_Ele27_WP80_DiCentralPFJet25_PFMHT15_v6', 
    'HLT_Ele27_WP80_DiCentralPFJet25_v6', 
    'HLT_Ele27_WP80_DiPFJet25_Deta3_v6', 
    'HLT_Ele32_WP80_DiCentralPFJet25_PFMHT25_v6', 
    'HLT_Ele32_WP80_DiPFJet25_Deta3p5_v6', 
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v10', 
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v10', 
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v10', 
    'HLT_HT350_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT45_v12', 
    'HLT_HT400_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT50_v6', 
    'HLT_HT400_Ele60_CaloIdT_TrkIdT_v6', 
    'HLT_HT450_Ele60_CaloIdT_TrkIdT_v5')

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
streamA_datasetHT_selector.triggerConditions = cms.vstring('HLT_DiJet130_PT130_v10', 
    'HLT_DiJet160_PT160_v10', 
    'HLT_FatJetMass850_DR1p1_Deta2p0_v6', 
    'HLT_HT150_v12', 
    'HLT_HT2000_v6', 
    'HLT_HT200_v12', 
    'HLT_HT250_AlphaT0p58_v4', 
    'HLT_HT250_AlphaT0p60_v4', 
    'HLT_HT250_AlphaT0p65_v3', 
    'HLT_HT250_L1FastJet_DoubleDisplacedPFJet60_ChgFraction10_v1', 
    'HLT_HT250_L1FastJet_DoubleDisplacedPFJet60_v1', 
    'HLT_HT250_L1FastJet_SingleDisplacedPFJet60_ChgFraction10_v1', 
    'HLT_HT250_L1FastJet_SingleDisplacedPFJet60_v1', 
    'HLT_HT250_v12', 
    'HLT_HT300_AlphaT0p54_v6', 
    'HLT_HT300_AlphaT0p55_v4', 
    'HLT_HT300_AlphaT0p60_v3', 
    'HLT_HT300_v13', 
    'HLT_HT350_AlphaT0p53_v11', 
    'HLT_HT350_L1FastJet_MHT100_v2', 
    'HLT_HT350_L1FastJet_MHT110_v2', 
    'HLT_HT350_L1FastJet_v4', 
    'HLT_HT350_MHT100_v4', 
    'HLT_HT350_MHT110_v4', 
    'HLT_HT350_v12', 
    'HLT_HT400_AlphaT0p51_v11', 
    'HLT_HT400_AlphaT0p52_v6', 
    'HLT_HT400_L1FastJet_MHT100_v2', 
    'HLT_HT400_L1FastJet_MHT90_v2', 
    'HLT_HT400_L1FastJet_v4', 
    'HLT_HT400_MHT100_v4', 
    'HLT_HT400_MHT90_v4', 
    'HLT_HT400_v12', 
    'HLT_HT450_AlphaT0p51_v6', 
    'HLT_HT450_v12', 
    'HLT_HT500_v12', 
    'HLT_HT550_v12', 
    'HLT_HT600_v5', 
    'HLT_HT650_v5', 
    'HLT_HT700_v3', 
    'HLT_HT750_L1FastJet_v4', 
    'HLT_HT750_v4', 
    'HLT_PFHT350_PFMHT100_v2', 
    'HLT_PFHT350_PFMHT90_v2', 
    'HLT_PFHT400_PFMHT80_v2', 
    'HLT_PFHT400_PFMHT90_v2', 
    'HLT_PFHT650_v2', 
    'HLT_R014_MR150_v11', 
    'HLT_R020_MR150_v11', 
    'HLT_R020_MR550_v11', 
    'HLT_R025_MR150_v11', 
    'HLT_R025_MR450_v11', 
    'HLT_R033_MR350_v11', 
    'HLT_R038_MR250_v11', 
    'HLT_R038_MR300_v3', 
    'HLT_RMR65_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v6', 
    'HLT_L1Tech_HBHEHO_totalOR_v4', 
    'HLT_L1Tech_HCAL_HF_single_channel_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v8', 
    'HLT_HcalPhiSym_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighPileUp_selector
streamA_datasetHighPileUp_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighPileUp_selector.l1tResults = cms.InputTag('')
streamA_datasetHighPileUp_selector.throw      = cms.bool(False)
streamA_datasetHighPileUp_selector.triggerConditions = cms.vstring('HLT_300Tower0p5_v2', 
    'HLT_300Tower0p6_v2', 
    'HLT_300Tower0p7_v2', 
    'HLT_300Tower0p8_v2', 
    'HLT_70Jet10_v5', 
    'HLT_70Jet13_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJet_selector
streamA_datasetJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJet_selector.l1tResults = cms.InputTag('')
streamA_datasetJet_selector.throw      = cms.bool(False)
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiJetAve110_v10', 
    'HLT_DiJetAve190_v10', 
    'HLT_DiJetAve240_v10', 
    'HLT_DiJetAve300_v11', 
    'HLT_DiJetAve30_v10', 
    'HLT_DiJetAve370_v11', 
    'HLT_DiJetAve60_v10', 
    'HLT_Jet110_v10', 
    'HLT_Jet190_v10', 
    'HLT_Jet240_L1FastJet_v4', 
    'HLT_Jet240_v10', 
    'HLT_Jet300_L1FastJet_v4', 
    'HLT_Jet300_v10', 
    'HLT_Jet30_L1FastJet_v4', 
    'HLT_Jet30_v10', 
    'HLT_Jet370_L1FastJet_v4', 
    'HLT_Jet370_NoJetID_v11', 
    'HLT_Jet370_v11', 
    'HLT_Jet60_L1FastJet_v4', 
    'HLT_Jet60_v10', 
    'HLT_Jet800_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetLogMonitor_selector
streamA_datasetLogMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetLogMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetLogMonitor_selector.throw      = cms.bool(False)
streamA_datasetLogMonitor_selector.triggerConditions = cms.vstring('HLT_LogMonitor_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMET_selector
streamA_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetMET_selector.throw      = cms.bool(False)
streamA_datasetMET_selector.triggerConditions = cms.vstring('HLT_CentralJet80_MET110_v4', 
    'HLT_CentralJet80_MET65_v11', 
    'HLT_CentralJet80_MET80_v10', 
    'HLT_CentralJet80_MET95_v4', 
    'HLT_DiCentralJet20_BTagIP_MET65_v12', 
    'HLT_DiCentralJet20_MET100_HBHENoiseFiltered_v5', 
    'HLT_DiCentralJet20_MET80_v9', 
    'HLT_DiCentralPFJet30_PFMHT80_v2', 
    'HLT_DiCentralPFJet50_PFMHT80_v2', 
    'HLT_DiJet60_MET45_v11', 
    'HLT_IsoMu15_eta2p1_L1ETM20_v1', 
    'HLT_L2Mu60_1Hit_MET40_v7', 
    'HLT_L2Mu60_1Hit_MET60_v7', 
    'HLT_MET120_HBHENoiseFiltered_v7', 
    'HLT_MET120_v8', 
    'HLT_MET200_HBHENoiseFiltered_v7', 
    'HLT_MET200_v8', 
    'HLT_MET400_v3', 
    'HLT_Mu15_eta2p1_L1ETM20_v1', 
    'HLT_PFMHT150_v18', 
    'HLT_R014_MR200_CentralJet40_BTagIP_v5', 
    'HLT_R014_MR400_CentralJet40_BTagIP_v5', 
    'HLT_R014_MR450_CentralJet40_BTagIP_v5', 
    'HLT_R020_MR300_CentralJet40_BTagIP_v5', 
    'HLT_R020_MR350_CentralJet40_BTagIP_v5', 
    'HLT_R030_MR200_CentralJet40_BTagIP_v5', 
    'HLT_R030_MR250_CentralJet40_BTagIP_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v11', 
    'HLT_JetE30_NoBPTX_v9', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v6', 
    'HLT_JetE70_NoBPTX3BX_NoHalo_v1', 
    'HLT_Physics_v3', 
    'HLT_PixelTracks_Multiplicity100_v9', 
    'HLT_PixelTracks_Multiplicity80_v9', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_Ele8_CaloIdT_TrkIdT_v5', 
    'HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v9', 
    'HLT_Mu15_DoublePhoton15_CaloIdL_v15', 
    'HLT_Mu15_Photon20_CaloIdL_v15', 
    'HLT_Mu17_Ele8_CaloIdL_v14', 
    'HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_v9', 
    'HLT_Mu5_DoubleEle8_CaloIdT_TrkIdVL_v9', 
    'HLT_Mu5_Ele8_CaloIdT_CaloIsoVL_v6', 
    'HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v9', 
    'HLT_Mu8_Ele17_CaloIdL_v14', 
    'HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_v9', 
    'HLT_Mu8_Photon20_CaloIdVT_IsoT_v14')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleDisplacedMu4_DiPFJet40Neutral_L1FastJet_v1', 
    'HLT_DoubleMu5_Mass8_HT150_v6', 
    'HLT_DoubleMu8_Mass8_HT150_v6', 
    'HLT_DoubleMu8_Mass8_HT200_v6', 
    'HLT_DoubleTkIso10Mu5_Mass8_HT150_v6', 
    'HLT_DoubleTkIso10Mu5_Mass8_HT200_v6', 
    'HLT_HT300_Mu15_PFMHT40_v7', 
    'HLT_HT300_Mu15_PFMHT50_v6', 
    'HLT_HT350_Mu5_PFMHT45_v14', 
    'HLT_HT400_Mu5_PFMHT50_v6', 
    'HLT_Iso10Mu20_eta2p1_CentralPFJet30_BTagIPIter_v1', 
    'HLT_Iso10Mu20_eta2p1_CentralPFJet30_v1', 
    'HLT_Iso10Mu20_eta2p1_DiCentralPFJet30_v1', 
    'HLT_Iso10Mu20_eta2p1_QuadCentralPFJet30_v1', 
    'HLT_Iso10Mu20_eta2p1_TriCentralPFJet30_v1', 
    'HLT_IsoMu17_eta2p1_DiCentralPFJet25_PFMHT15_v6', 
    'HLT_IsoMu17_eta2p1_DiCentralPFJet25_PFMHT25_v6', 
    'HLT_IsoMu17_eta2p1_DiCentralPFJet25_v6', 
    'HLT_IsoMu17_eta2p1_DiPFJet25_Deta3_PFJet25_v6', 
    'HLT_IsoMu17_eta2p1_DiPFJet25_Deta3_v6', 
    'HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_L1FastJet_v1', 
    'HLT_Mu10_R014_MR200_v6', 
    'HLT_Mu10_R025_MR200_v7', 
    'HLT_Mu10_R029_MR200_v7', 
    'HLT_Mu10_R033_MR200_v6', 
    'HLT_Mu12_eta2p1_DiCentralJet20_BTagIP3D1stTrack_v7', 
    'HLT_Mu12_eta2p1_DiCentralJet20_DiBTagIP3D1stTrack_v7', 
    'HLT_Mu17_eta2p1_DiCentralPFJet25_PFMHT15_v6', 
    'HLT_Mu17_eta2p1_DiPFJet25_Deta3_v6', 
    'HLT_Mu20_eta2p1_QuadCentralPFJet30_v1', 
    'HLT_Mu20_eta2p1_TriCentralPFJet30_v1', 
    'HLT_Mu40_HT300_v6', 
    'HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_HT150_v6', 
    'HLT_Mu60_HT300_v6', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_HT150_v6', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_HT200_v6', 
    'HLT_TkIso10Mu5_Ele8_CaloIdT_CaloIsoVVL_TrkIdVL_Mass8_HT150_v6', 
    'HLT_TkIso10Mu5_Ele8_CaloIdT_CaloIsoVVL_TrkIdVL_Mass8_HT200_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_Muon_v12', 
    'HLT_Dimuon0_Jpsi_NoVertexing_v8', 
    'HLT_Dimuon0_Jpsi_v11', 
    'HLT_Dimuon0_PsiPrime_v1', 
    'HLT_Dimuon0_Upsilon_Muon_v12', 
    'HLT_Dimuon0_Upsilon_v11', 
    'HLT_Dimuon11_PsiPrime_v6', 
    'HLT_Dimuon3p5_SameSign_v1', 
    'HLT_Dimuon5_Jpsi_v1', 
    'HLT_Dimuon5_PsiPrime_v1', 
    'HLT_Dimuon5_Upsilon_v1', 
    'HLT_Dimuon6_LowMass_v6', 
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
    'HLT_DoubleJet30_ForwardBackward_v11', 
    'HLT_DoubleJet60_ForwardBackward_v11', 
    'HLT_DoubleJet70_ForwardBackward_v11', 
    'HLT_DoubleJet80_ForwardBackward_v11', 
    'HLT_EightJet120_v6', 
    'HLT_EightJet35_L1FastJet_v3', 
    'HLT_EightJet35_v4', 
    'HLT_EightJet40_L1FastJet_v3', 
    'HLT_EightJet40_v4', 
    'HLT_ExclDiJet80_HFAND_v1', 
    'HLT_Jet160Eta2p4_Jet120Eta2p4_L1FastJet_DiBTagIP3DLoose_v1', 
    'HLT_Jet60Eta1p7_Jet53Eta1p7_L1FastJet_DiBTagIP3D_v1', 
    'HLT_Jet80Eta1p7_Jet70Eta1p7_L1FastJet_DiBTagIP3D_v1', 
    'HLT_L1DoubleJet36Central_v5', 
    'HLT_QuadJet40_IsoPFTau40_v19', 
    'HLT_QuadJet40_v12', 
    'HLT_QuadJet45_DiJet40_v4', 
    'HLT_QuadJet45_IsoPFTau45_v14', 
    'HLT_QuadJet50_DiJet40_L1FastJet_v3', 
    'HLT_QuadJet50_DiJet40_v6', 
    'HLT_QuadJet50_IsoPFTau50_v8', 
    'HLT_QuadJet70_v11', 
    'HLT_QuadJet80_L1FastJet_v3', 
    'HLT_QuadJet80_v6', 
    'HLT_QuadJet90_v4', 
    'HLT_QuadL1FastJet_BTagIP_VBF_v1', 
    'HLT_SixJet45_L1FastJet_v3', 
    'HLT_SixJet45_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoubleEle33_CaloIdL_CaloIsoT_v4', 
    'HLT_DoubleEle33_CaloIdL_v8', 
    'HLT_DoubleEle33_CaloIdT_v4', 
    'HLT_DoubleEle45_CaloIdL_v7', 
    'HLT_DoublePhoton40_CaloIdL_MR150_v4', 
    'HLT_DoublePhoton40_CaloIdL_R014_MR150_v4', 
    'HLT_DoublePhoton43_HEVT_v2', 
    'HLT_DoublePhoton48_HEVT_v2', 
    'HLT_DoublePhoton5_IsoVL_CEP_v10', 
    'HLT_DoublePhoton70_v2', 
    'HLT_DoublePhoton80_v3', 
    'HLT_Photon135_v3', 
    'HLT_Photon200_NoHE_v5', 
    'HLT_Photon20_CaloIdVL_IsoL_v10', 
    'HLT_Photon20_R9Id_Photon18_R9Id_v8', 
    'HLT_Photon225_NoHE_v3', 
    'HLT_Photon26_CaloIdXL_IsoXL_Photon18_CaloIdXL_IsoXL_Mass60_v5', 
    'HLT_Photon26_CaloIdXL_IsoXL_Photon18_R9IdT_Mass60_v5', 
    'HLT_Photon26_CaloIdXL_IsoXL_Photon18_v5', 
    'HLT_Photon26_Photon18_v8', 
    'HLT_Photon26_R9IdT_Photon18_CaloIdXL_IsoXL_Mass60_v5', 
    'HLT_Photon26_R9IdT_Photon18_R9IdT_Mass60_v2', 
    'HLT_Photon30_CaloIdVL_IsoL_v12', 
    'HLT_Photon30_CaloIdVL_v9', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_CaloIdL_IsoVL_v8', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_R9Id_v7', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_v9', 
    'HLT_Photon36_CaloIdVL_Photon22_CaloIdVL_v3', 
    'HLT_Photon36_Photon22_v2', 
    'HLT_Photon36_R9Id_Photon22_CaloIdL_IsoVL_v8', 
    'HLT_Photon36_R9Id_Photon22_R9Id_v4', 
    'HLT_Photon400_v3', 
    'HLT_Photon50_CaloIdVL_IsoL_v10', 
    'HLT_Photon50_CaloIdVL_v5', 
    'HLT_Photon75_CaloIdVL_IsoL_v11', 
    'HLT_Photon75_CaloIdVL_v8', 
    'HLT_Photon90_CaloIdVL_IsoL_v8', 
    'HLT_Photon90_CaloIdVL_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon30_CaloIdVT_CentralJet20_BTagIP_v8', 
    'HLT_Photon40_CaloIdL_R014_MR150_v2', 
    'HLT_Photon40_CaloIdL_R017_MR500_v7', 
    'HLT_Photon40_CaloIdL_R023_MR350_v7', 
    'HLT_Photon40_CaloIdL_R029_MR250_v7', 
    'HLT_Photon40_CaloIdL_R042_MR200_v7', 
    'HLT_Photon55_CaloIdL_R017_MR500_v5', 
    'HLT_Photon55_CaloIdL_R023_MR350_v5', 
    'HLT_Photon55_CaloIdL_R029_MR250_v5', 
    'HLT_Photon55_CaloIdL_R042_MR200_v5', 
    'HLT_Photon60_CaloIdL_HT300_v4', 
    'HLT_Photon60_CaloIdL_MHT70_v4', 
    'HLT_Photon70_CaloIdXL_HT400_v4', 
    'HLT_Photon70_CaloIdXL_HT500_v4', 
    'HLT_Photon70_CaloIdXL_MHT100_v4', 
    'HLT_Photon70_CaloIdXL_MHT90_v4', 
    'HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet25_v6', 
    'HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet30_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele100_CaloIdVT_TrkIdT_v4', 
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v4', 
    'HLT_Ele27_WP80_PFMT50_v10', 
    'HLT_Ele27_WP80_v4', 
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
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v1', 
    'HLT_L2Mu20_NoVertex_NoBPTX3BX_NoHalo_v1', 
    'HLT_L2Mu20_eta2p1_NoVertex_v1', 
    'HLT_L2Mu30_NoVertex_NoBPTX3BX_NoHalo_v1', 
    'HLT_Mu12_v13', 
    'HLT_Mu15_eta2p1_v1', 
    'HLT_Mu17_v1', 
    'HLT_Mu24_eta2p1_v1', 
    'HLT_Mu30_eta2p1_v1', 
    'HLT_Mu40_eta2p1_v6', 
    'HLT_Mu50_eta2p1_v3', 
    'HLT_Mu5_v15', 
    'HLT_Mu8_v13')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleIsoPFTau45_Trk5_eta2p1_v9', 
    'HLT_DoubleIsoPFTau55_Trk5_eta2p1_v6', 
    'HLT_MediumIsoPFTau35_Trk20_MET60_v7', 
    'HLT_MediumIsoPFTau35_Trk20_MET70_v7', 
    'HLT_MediumIsoPFTau35_Trk20_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele18_CaloIdVT_TrkIdT_MediumIsoPFTau20_v7', 
    'HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v7', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau25_v6', 
    'HLT_HT350_DoubleIsoPFTau10_Trk3_PFMHT45_v14', 
    'HLT_HT400_DoubleIsoPFTau10_Trk3_PFMHT50_v6', 
    'HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v7', 
    'HLT_IsoMu15_eta2p1_MediumIsoPFTau20_v7', 
    'HLT_IsoMu15_eta2p1_TightIsoPFTau20_v7', 
    'HLT_Mu15_LooseIsoPFTau15_v15')

