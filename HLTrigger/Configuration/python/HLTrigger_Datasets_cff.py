# /dev/CMSSW_4_2_0/GRun/V160

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTag_selector
streamA_datasetBTag_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTag_selector.l1tResults = cms.InputTag('')
streamA_datasetBTag_selector.throw      = cms.bool(False)
streamA_datasetBTag_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet110_Mu5_v7', 
    'HLT_BTagMu_DiJet20_Mu5_v7', 
    'HLT_BTagMu_DiJet40_Mu5_v7', 
    'HLT_BTagMu_DiJet70_Mu5_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v7', 
    'HLT_BeamGas_BSC_v4', 
    'HLT_BeamGas_HF_v6', 
    'HLT_IsoTrackHB_v5', 
    'HLT_IsoTrackHE_v6', 
    'HLT_L1SingleEG12_v3', 
    'HLT_L1SingleEG5_v3', 
    'HLT_L1SingleJet16_v4', 
    'HLT_L1SingleJet36_v4', 
    'HLT_L1SingleMuOpen_DT_v4', 
    'HLT_L1SingleMuOpen_v4', 
    'HLT_L1_Interbunch_BSC_v3', 
    'HLT_L1_PreCollisions_v3', 
    'HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v4', 
    'HLT_L1SingleMuOpen_AntiBPTX_v3', 
    'HLT_L1TrackerCosmics_v4', 
    'HLT_RegionalCosmicTracking_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v1', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele15_HFL_v8', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele15_HFT_v3', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele8_CaloIdL_CaloIsoVL_v6', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v6', 
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v5', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass30_v4', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v6', 
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_v4', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v6', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v6', 
    'HLT_Ele8_CaloIdL_TrkIdVL_v6', 
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v4', 
    'HLT_Ele8_v6', 
    'HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v6', 
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_v7', 
    'HLT_DoubleMu45_v3', 
    'HLT_DoubleMu4_Acoplanarity03_v6', 
    'HLT_DoubleMu5_Acoplanarity03_v3', 
    'HLT_DoubleMu5_IsoMu5_v5', 
    'HLT_DoubleMu6_v5', 
    'HLT_DoubleMu7_v5', 
    'HLT_L1DoubleMu0_v4', 
    'HLT_L2DoubleMu0_v6', 
    'HLT_L2DoubleMu23_NoVertex_v5', 
    'HLT_L2DoubleMu30_NoVertex_v1', 
    'HLT_Mu13_Mu8_v4', 
    'HLT_Mu17_Mu8_v4', 
    'HLT_Mu8_Jet40_v7', 
    'HLT_TripleMu5_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_DoubleEle8_CaloIdT_TrkIdVL_HT150_v4', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass4_HT150_v1', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_v1', 
    'HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R005_MR200_v3', 
    'HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R025_MR200_v3', 
    'HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R029_MR200_v1', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT25_v1', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_v6', 
    'HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_v1', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta3_Jet20_v1', 
    'HLT_Ele15_CaloIdVT_TrkIdT_Jet35_Jet25_Deta3_Jet20_v2', 
    'HLT_Ele17_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta3p5_Jet25_v2', 
    'HLT_Ele22_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_CentralJet25_PFMHT20_v2', 
    'HLT_Ele22_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_CentralJet25_v2', 
    'HLT_Ele22_CaloIdVT_TrkIdT_CentralJet30_CentralJet25_v2', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_BTagIP_v2', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_v2', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralJet30_PFMHT25_v2', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralJet30_v2', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_QuadCentralJet30_v2', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralJet30_v2', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_BTagIP_v6', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_v6', 
    'HLT_Ele25_CaloIdVT_TrkIdT_DiCentralJet30_v5', 
    'HLT_Ele25_CaloIdVT_TrkIdT_QuadCentralJet30_v2', 
    'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralJet30_v5', 
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v3', 
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v3', 
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v3', 
    'HLT_HT200_DoubleEle5_CaloIdVL_MassJPsi_v1', 
    'HLT_HT300_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT40_v3', 
    'HLT_HT350_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT45_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFEDMonitor_selector
streamA_datasetFEDMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFEDMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetFEDMonitor_selector.throw      = cms.bool(False)
streamA_datasetFEDMonitor_selector.triggerConditions = cms.vstring('HLT_DTErrors_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHT_selector
streamA_datasetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHT_selector.throw      = cms.bool(False)
streamA_datasetHT_selector.triggerConditions = cms.vstring('HLT_DiJet130_PT130_v5', 
    'HLT_DiJet160_PT160_v5', 
    'HLT_FatJetMass300_DR1p1_Deta2p0_CentralJet30_BTagIP_v1', 
    'HLT_FatJetMass350_DR1p1_Deta2p0_CentralJet30_BTagIP_v1', 
    'HLT_FatJetMass750_DR1p1_Deta2p0_v1', 
    'HLT_FatJetMass850_DR1p1_Deta2p0_v1', 
    'HLT_HT150_v7', 
    'HLT_HT2000_v1', 
    'HLT_HT200_AlphaT0p55_v1', 
    'HLT_HT200_v7', 
    'HLT_HT250_AlphaT0p53_v5', 
    'HLT_HT250_AlphaT0p55_v1', 
    'HLT_HT250_DoubleDisplacedJet60_PromptTrack_v4', 
    'HLT_HT250_DoubleDisplacedJet60_v6', 
    'HLT_HT250_MHT100_v1', 
    'HLT_HT250_MHT90_v1', 
    'HLT_HT250_v7', 
    'HLT_HT300_AlphaT0p53_v5', 
    'HLT_HT300_AlphaT0p54_v1', 
    'HLT_HT300_CentralJet30_BTagIP_PFMHT55_v5', 
    'HLT_HT300_CentralJet30_BTagIP_PFMHT75_v5', 
    'HLT_HT300_CentralJet30_BTagIP_v5', 
    'HLT_HT300_MHT80_v1', 
    'HLT_HT300_MHT90_v1', 
    'HLT_HT300_PFMHT55_v5', 
    'HLT_HT300_v8', 
    'HLT_HT350_AlphaT0p52_v1', 
    'HLT_HT350_AlphaT0p53_v6', 
    'HLT_HT350_MHT70_v1', 
    'HLT_HT350_MHT80_v1', 
    'HLT_HT350_v7', 
    'HLT_HT400_AlphaT0p51_v6', 
    'HLT_HT400_AlphaT0p52_v1', 
    'HLT_HT400_v7', 
    'HLT_HT450_AlphaT0p51_v1', 
    'HLT_HT450_AlphaT0p52_v1', 
    'HLT_HT450_JetPt60_DPhi2p94_v1', 
    'HLT_HT450_v7', 
    'HLT_HT500_JetPt60_DPhi2p94_v1', 
    'HLT_HT500_v7', 
    'HLT_HT550_JetPt60_DPhi2p94_v1', 
    'HLT_HT550_v7', 
    'HLT_HT600_v1', 
    'HLT_R014_MR150_CentralJet40_BTagIP_v6', 
    'HLT_R014_MR150_v5', 
    'HLT_R017_MR450_CentralJet40_BTagIP_v1', 
    'HLT_R017_MR500_CentralJet40_BTagIP_v1', 
    'HLT_R020_MR150_v5', 
    'HLT_R020_MR550_v5', 
    'HLT_R023_MR350_CentralJet40_BTagIP_v1', 
    'HLT_R023_MR400_CentralJet40_BTagIP_v1', 
    'HLT_R023_MR550_v1', 
    'HLT_R025_MR150_v5', 
    'HLT_R025_MR450_v5', 
    'HLT_R029_MR250_CentralJet40_BTagIP_v1', 
    'HLT_R029_MR300_CentralJet40_BTagIP_v1', 
    'HLT_R029_MR450_v1', 
    'HLT_R033_MR200_CentralJet40_BTagIP_v1', 
    'HLT_R033_MR350_v5', 
    'HLT_R036_MR200_CentralJet40_BTagIP_v1', 
    'HLT_R036_MR350_v1', 
    'HLT_R038_MR250_v5', 
    'HLT_R042_MR250_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v4', 
    'HLT_L1Tech_HBHEHO_totalOR_v3', 
    'HLT_L1Tech_HCAL_HF_single_channel_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v6', 
    'HLT_HcalPhiSym_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJet_selector
streamA_datasetJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJet_selector.l1tResults = cms.InputTag('')
streamA_datasetJet_selector.throw      = cms.bool(False)
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiJetAve110_v6', 
    'HLT_DiJetAve150_v6', 
    'HLT_DiJetAve190_v6', 
    'HLT_DiJetAve240_v6', 
    'HLT_DiJetAve300_v6', 
    'HLT_DiJetAve30_v6', 
    'HLT_DiJetAve370_v6', 
    'HLT_DiJetAve60_v6', 
    'HLT_DiJetAve80_v6', 
    'HLT_Jet110_v6', 
    'HLT_Jet150_v6', 
    'HLT_Jet190_v6', 
    'HLT_Jet240_CentralJet30_BTagIP_v1', 
    'HLT_Jet240_v6', 
    'HLT_Jet270_CentralJet30_BTagIP_v1', 
    'HLT_Jet300_v5', 
    'HLT_Jet30_v6', 
    'HLT_Jet370_NoJetID_v6', 
    'HLT_Jet370_v6', 
    'HLT_Jet60_v6', 
    'HLT_Jet800_v1', 
    'HLT_Jet80_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetLogMonitor_selector
streamA_datasetLogMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetLogMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetLogMonitor_selector.throw      = cms.bool(False)
streamA_datasetLogMonitor_selector.triggerConditions = cms.vstring('HLT_LogMonitor_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMET_selector
streamA_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetMET_selector.throw      = cms.bool(False)
streamA_datasetMET_selector.triggerConditions = cms.vstring('HLT_CentralJet80_MET100_v6', 
    'HLT_CentralJet80_MET160_v6', 
    'HLT_CentralJet80_MET65_v6', 
    'HLT_CentralJet80_MET80_v6', 
    'HLT_DiCentralJet20_BTagIP_MET65_v5', 
    'HLT_DiCentralJet20_MET80_v4', 
    'HLT_DiJet60_MET45_v6', 
    'HLT_L2Mu60_1Hit_MET40_v3', 
    'HLT_L2Mu60_1Hit_MET60_v3', 
    'HLT_MET100_HBHENoiseFiltered_v4', 
    'HLT_MET100_v6', 
    'HLT_MET120_HBHENoiseFiltered_v4', 
    'HLT_MET120_v6', 
    'HLT_MET200_HBHENoiseFiltered_v4', 
    'HLT_MET200_v6', 
    'HLT_MET400_v1', 
    'HLT_MET65_HBHENoiseFiltered_v3', 
    'HLT_MET65_v3', 
    'HLT_PFMHT150_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v7', 
    'HLT_JetE30_NoBPTX_NoHalo_v7', 
    'HLT_JetE30_NoBPTX_v5', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v3', 
    'HLT_Physics_v2', 
    'HLT_PixelTracks_Multiplicity100_v5', 
    'HLT_PixelTracks_Multiplicity80_v5', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_Ele8_CaloIdL_TrkIdVL_v7', 
    'HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v1', 
    'HLT_Mu15_DoublePhoton15_CaloIdL_v7', 
    'HLT_Mu15_Photon20_CaloIdL_v7', 
    'HLT_Mu17_Ele8_CaloIdL_v6', 
    'HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_v1', 
    'HLT_Mu3_Ele8_CaloIdT_CaloIsoVL_v1', 
    'HLT_Mu5_DoubleEle8_CaloIdT_TrkIdVL_v1', 
    'HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v1', 
    'HLT_Mu8_Ele17_CaloIdL_v6', 
    'HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_v1', 
    'HLT_Mu8_Photon20_CaloIdVT_IsoT_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_HT150_v5', 
    'HLT_DoubleMu3_HT200_v8', 
    'HLT_DoubleMu3_Mass4_HT150_v1', 
    'HLT_HT250_Mu15_PFMHT20_v5', 
    'HLT_HT250_Mu15_PFMHT40_v1', 
    'HLT_HT300_Mu5_PFMHT40_v5', 
    'HLT_HT350_Mu5_PFMHT45_v5', 
    'HLT_IsoMu17_CentralJet30_BTagIP_v7', 
    'HLT_IsoMu17_CentralJet30_v3', 
    'HLT_IsoMu17_DiCentralJet30_v3', 
    'HLT_IsoMu17_QuadCentralJet30_v3', 
    'HLT_IsoMu17_TriCentralJet30_v3', 
    'HLT_IsoMu20_DiCentralJet34_v1', 
    'HLT_Mu12_DiCentralJet20_BTagIP3D1stTrack_v1', 
    'HLT_Mu12_DiCentralJet20_DiBTagIP3D1stTrack_v1', 
    'HLT_Mu12_DiCentralJet30_BTagIP3D_v3', 
    'HLT_Mu17_CentralJet30_BTagIP_v7', 
    'HLT_Mu17_CentralJet30_v8', 
    'HLT_Mu17_DiCentralJet30_v8', 
    'HLT_Mu17_QuadCentralJet30_v3', 
    'HLT_Mu17_TriCentralJet30_v8', 
    'HLT_Mu30_HT200_v1', 
    'HLT_Mu3_DiJet30_v4', 
    'HLT_Mu3_Ele8_CaloIdT_TrkIdVL_HT150_v4', 
    'HLT_Mu3_QuadJet30_v4', 
    'HLT_Mu3_TriJet30_v4', 
    'HLT_Mu40_HT200_v1', 
    'HLT_Mu8_R005_MR200_v5', 
    'HLT_Mu8_R025_MR200_v5', 
    'HLT_Mu8_R029_MR200_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_Muon_v4', 
    'HLT_Dimuon0_Jpsi_NoVertexing_v1', 
    'HLT_Dimuon0_Jpsi_v3', 
    'HLT_Dimuon0_Upsilon_Muon_v4', 
    'HLT_Dimuon0_Upsilon_v3', 
    'HLT_Dimuon10_Jpsi_Barrel_v3', 
    'HLT_Dimuon4_Bs_Barrel_v5', 
    'HLT_Dimuon5_Upsilon_Barrel_v3', 
    'HLT_Dimuon6_Bs_v4', 
    'HLT_Dimuon7_Jpsi_X_Barrel_v3', 
    'HLT_Dimuon7_PsiPrime_v3', 
    'HLT_DoubleMu3p5_Jpsi_Displaced_v1', 
    'HLT_DoubleMu4_LowMass_Displaced_v1', 
    'HLT_Mu5_L2Mu2_Jpsi_v6', 
    'HLT_Mu5_Track2_Jpsi_v6', 
    'HLT_Mu7_Track7_Jpsi_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_CentralJet46_CentralJet38_DiBTagIP3D_v1', 
    'HLT_CentralJet60_CentralJet53_DiBTagIP3D_v1', 
    'HLT_DoubleJet30_ForwardBackward_v7', 
    'HLT_DoubleJet60_ForwardBackward_v7', 
    'HLT_DoubleJet70_ForwardBackward_v7', 
    'HLT_DoubleJet80_ForwardBackward_v7', 
    'HLT_EightJet120_v1', 
    'HLT_ExclDiJet60_HFAND_v6', 
    'HLT_ExclDiJet60_HFOR_v6', 
    'HLT_L1DoubleJet36Central_v4', 
    'HLT_L1ETM30_v4', 
    'HLT_L1MultiJet_v4', 
    'HLT_QuadJet40_IsoPFTau40_v9', 
    'HLT_QuadJet40_v7', 
    'HLT_QuadJet45_IsoPFTau45_v4', 
    'HLT_QuadJet50_Jet40_Jet30_v3', 
    'HLT_QuadJet60_v6', 
    'HLT_QuadJet70_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoubleEle33_CaloIdL_v3', 
    'HLT_DoubleEle33_v3', 
    'HLT_DoubleEle45_CaloIdL_v2', 
    'HLT_DoublePhoton33_HEVT_v3', 
    'HLT_DoublePhoton38_HEVT_v2', 
    'HLT_DoublePhoton40_MR150_v4', 
    'HLT_DoublePhoton40_R014_MR150_v4', 
    'HLT_DoublePhoton5_IsoVL_CEP_v5', 
    'HLT_DoublePhoton60_v3', 
    'HLT_DoublePhoton80_v1', 
    'HLT_Photon135_v1', 
    'HLT_Photon200_NoHE_v3', 
    'HLT_Photon20_CaloIdVL_IsoL_v5', 
    'HLT_Photon20_R9Id_Photon18_R9Id_v6', 
    'HLT_Photon225_NoHE_v1', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_CaloIdL_IsoVL_v6', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_R9Id_v5', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_v6', 
    'HLT_Photon26_IsoVL_Photon18_IsoVL_v6', 
    'HLT_Photon26_IsoVL_Photon18_v6', 
    'HLT_Photon26_Photon18_v6', 
    'HLT_Photon26_R9Id_Photon18_CaloIdL_IsoVL_v5', 
    'HLT_Photon26_R9Id_Photon18_R9Id_v3', 
    'HLT_Photon30_CaloIdVL_IsoL_v6', 
    'HLT_Photon30_CaloIdVL_v6', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_CaloIdL_IsoVL_v2', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_CaloIdL_v2', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_R9Id_v1', 
    'HLT_Photon36_CaloIdL_IsoVL_Photon22_v3', 
    'HLT_Photon36_CaloIdL_Photon22_CaloIdL_v5', 
    'HLT_Photon36_CaloIdVL_Photon22_CaloIdVL_v1', 
    'HLT_Photon36_IsoVL_Photon22_v3', 
    'HLT_Photon36_R9Id_Photon22_CaloIdL_IsoVL_v2', 
    'HLT_Photon36_R9Id_Photon22_R9Id_v2', 
    'HLT_Photon400_v1', 
    'HLT_Photon40_CaloIdL_Photon28_CaloIdL_v3', 
    'HLT_Photon44_CaloIdL_Photon34_CaloIdL_v1', 
    'HLT_Photon48_CaloIdL_Photon38_CaloIdL_v1', 
    'HLT_Photon50_CaloIdVL_IsoL_v5', 
    'HLT_Photon50_CaloIdVL_v3', 
    'HLT_Photon75_CaloIdVL_IsoL_v6', 
    'HLT_Photon75_CaloIdVL_v6', 
    'HLT_Photon90_CaloIdVL_IsoL_v3', 
    'HLT_Photon90_CaloIdVL_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon30_CaloIdVT_CentralJet20_BTagIP_v1', 
    'HLT_Photon40_CaloIdL_R005_MR150_v3', 
    'HLT_Photon40_CaloIdL_R014_MR500_v1', 
    'HLT_Photon40_CaloIdL_R017_MR500_v1', 
    'HLT_Photon40_CaloIdL_R020_MR350_v1', 
    'HLT_Photon40_CaloIdL_R023_MR350_v1', 
    'HLT_Photon40_CaloIdL_R025_MR250_v1', 
    'HLT_Photon40_CaloIdL_R029_MR250_v1', 
    'HLT_Photon40_CaloIdL_R038_MR200_v1', 
    'HLT_Photon40_CaloIdL_R042_MR200_v1', 
    'HLT_Photon70_CaloIdL_HT350_v6', 
    'HLT_Photon70_CaloIdL_HT400_v1', 
    'HLT_Photon70_CaloIdL_MHT70_v6', 
    'HLT_Photon70_CaloIdL_MHT90_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele100_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_v1', 
    'HLT_Ele25_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v2', 
    'HLT_Ele27_WP80_PFMT50_v1', 
    'HLT_Ele32_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_v3', 
    'HLT_Ele32_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v5', 
    'HLT_Ele32_WP70_PFMT50_v1', 
    'HLT_Ele52_CaloIdVT_TrkIdT_v3', 
    'HLT_Ele65_CaloIdVT_TrkIdT_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_IsoMu12_v7', 
    'HLT_IsoMu15_v11', 
    'HLT_IsoMu17_eta2p1_v1', 
    'HLT_IsoMu17_v11', 
    'HLT_IsoMu20_eta2p1_v1', 
    'HLT_IsoMu24_eta2p1_v1', 
    'HLT_IsoMu30_eta2p1_v1', 
    'HLT_L1SingleMu10_v4', 
    'HLT_L1SingleMu20_v4', 
    'HLT_L2Mu10_v5', 
    'HLT_L2Mu20_v5', 
    'HLT_Mu100_v3', 
    'HLT_Mu12_v5', 
    'HLT_Mu15_v6', 
    'HLT_Mu20_v5', 
    'HLT_Mu24_v5', 
    'HLT_Mu30_v5', 
    'HLT_Mu3_v7', 
    'HLT_Mu40_v3', 
    'HLT_Mu5_v7', 
    'HLT_Mu60_v1', 
    'HLT_Mu8_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleIsoPFTau45_Trk5_eta2p1_v1', 
    'HLT_IsoPFTau35_Trk20_MET60_v4', 
    'HLT_IsoPFTau35_Trk20_MET70_v1', 
    'HLT_IsoPFTau35_Trk20_v4', 
    'HLT_IsoPFTau40_IsoPFTau30_Trk5_eta2p1_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TightIsoPFTau20_v1', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v6', 
    'HLT_Ele15_CaloIdVT_TrkIdT_TightIsoPFTau20_v1', 
    'HLT_Ele18_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TightIsoPFTau20_v1', 
    'HLT_HT300_DoubleIsoPFTau10_Trk3_PFMHT40_v5', 
    'HLT_HT350_DoubleIsoPFTau10_Trk3_PFMHT45_v5', 
    'HLT_IsoMu15_LooseIsoPFTau15_v6', 
    'HLT_IsoMu15_LooseIsoPFTau20_v4', 
    'HLT_IsoMu15_TightIsoPFTau20_v4', 
    'HLT_Mu15_LooseIsoPFTau15_v6')

