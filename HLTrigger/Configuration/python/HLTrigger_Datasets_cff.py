# /online/collisions/2011/5e32/v6.2/HLT/V4

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v1', 
    'HLT_BeamGas_BSC_v2', 
    'HLT_BeamGas_HF_v2', 
    'HLT_IsoTrackHB_v2', 
    'HLT_IsoTrackHE_v3', 
    'HLT_L1SingleEG12_v1', 
    'HLT_L1SingleEG5_v1', 
    'HLT_L1SingleJet36_v1', 
    'HLT_L1SingleMuOpen_DT_v1', 
    'HLT_L1SingleMuOpen_v1', 
    'HLT_L1_Interbunch_BSC_v1', 
    'HLT_L1_PreCollisions_v1', 
    'HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v2', 
    'HLT_L1SingleMuOpen_AntiBPTX_v1', 
    'HLT_L1Tech_BSC_halo_v1', 
    'HLT_L1TrackerCosmics_v2', 
    'HLT_L3MuonsCosmicTracking_v1', 
    'HLT_RegionalCosmicTracking_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_v2', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele15_HFL_v2', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_Ele8_CaloIdL_CaloIsoVL_v2', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v2', 
    'HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v2', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v2', 
    'HLT_Ele32_CaloIdL_CaloIsoVL_SC17_v2', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v2', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v2', 
    'HLT_Ele8_CaloIdL_TrkIdVL_v2', 
    'HLT_Ele8_v2', 
    'HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v2', 
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_v3', 
    'HLT_DoubleMu4_Acoplanarity03_v1', 
    'HLT_DoubleMu6_v1', 
    'HLT_DoubleMu7_v1', 
    'HLT_L1DoubleMu0_v1', 
    'HLT_L2DoubleMu0_v2', 
    'HLT_L2DoubleMu23_NoVertex_v1', 
    'HLT_Mu8_Jet40_v3', 
    'HLT_TripleMu5_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_DoubleEle8_CaloIdL_TrkIdVL_HT160_v3', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_HT160_v3', 
    'HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT200_v3', 
    'HLT_Ele10_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT200_v3', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralDiJet30_v2', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_v2', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralJet40_BTagIP_v2', 
    'HLT_Ele25_CaloIdVT_TrkIdT_CentralTriJet30_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetForwardTriggers_selector
streamA_datasetForwardTriggers_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetForwardTriggers_selector.l1tResults = cms.InputTag('')
streamA_datasetForwardTriggers_selector.throw      = cms.bool(False)
streamA_datasetForwardTriggers_selector.triggerConditions = cms.vstring('HLT_L1Tech_CASTOR_HaloMuon_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHT_selector
streamA_datasetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHT_selector.throw      = cms.bool(False)
streamA_datasetHT_selector.triggerConditions = cms.vstring('HLT_HT150_AlphaT0p60_v1', 
    'HLT_HT150_AlphaT0p70_v1', 
    'HLT_HT150_v2', 
    'HLT_HT200_AlphaT0p60_v1', 
    'HLT_HT200_AlphaT0p65_v1', 
    'HLT_HT200_v2', 
    'HLT_HT250_AlphaT0p55_v1', 
    'HLT_HT250_AlphaT0p62_v1', 
    'HLT_HT250_DoubleDisplacedJet60_v1', 
    'HLT_HT250_MHT60_v2', 
    'HLT_HT250_v2', 
    'HLT_HT300_AlphaT0p52_v1', 
    'HLT_HT300_AlphaT0p54_v1', 
    'HLT_HT300_MHT75_v3', 
    'HLT_HT300_v3', 
    'HLT_HT350_AlphaT0p51_v1', 
    'HLT_HT350_AlphaT0p53_v1', 
    'HLT_HT350_v2', 
    'HLT_HT400_AlphaT0p51_v1', 
    'HLT_HT400_v2', 
    'HLT_HT450_v2', 
    'HLT_HT500_v2', 
    'HLT_HT550_v2', 
    'HLT_MR100_v1', 
    'HLT_Meff440_v2', 
    'HLT_Meff520_v2', 
    'HLT_Meff640_v2', 
    'HLT_R032_MR100_v1', 
    'HLT_R032_v1', 
    'HLT_R035_MR100_v1')

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
streamA_datasetJet_selector.triggerConditions = cms.vstring('HLT_DiJetAve100U_v4', 
    'HLT_DiJetAve140U_v4', 
    'HLT_DiJetAve15U_v4', 
    'HLT_DiJetAve180U_v4', 
    'HLT_DiJetAve300U_v4', 
    'HLT_DiJetAve30U_v4', 
    'HLT_DiJetAve50U_v4', 
    'HLT_DiJetAve70U_v4', 
    'HLT_Jet110_v1', 
    'HLT_Jet150_v1', 
    'HLT_Jet190_v1', 
    'HLT_Jet240_v1', 
    'HLT_Jet30_v1', 
    'HLT_Jet370_NoJetID_v1', 
    'HLT_Jet370_v1', 
    'HLT_Jet60_v1', 
    'HLT_Jet80_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMETBTag_selector
streamA_datasetMETBTag_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMETBTag_selector.l1tResults = cms.InputTag('')
streamA_datasetMETBTag_selector.throw      = cms.bool(False)
streamA_datasetMETBTag_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet100_Mu9_v2', 
    'HLT_BTagMu_DiJet20_Mu5_v2', 
    'HLT_BTagMu_DiJet60_Mu7_v2', 
    'HLT_BTagMu_DiJet80_Mu9_v2', 
    'HLT_CentralJet80_MET100_v1', 
    'HLT_CentralJet80_MET160_v1', 
    'HLT_CentralJet80_MET65_v1', 
    'HLT_CentralJet80_MET80_v1', 
    'HLT_DiJet60_MET45_v1', 
    'HLT_MET100_v1', 
    'HLT_MET120_v1', 
    'HLT_MET200_v1', 
    'HLT_PFMHT150_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v4', 
    'HLT_JetE30_NoBPTX_NoHalo_v4', 
    'HLT_JetE30_NoBPTX_v2', 
    'HLT_L1Tech_BSC_minBias_threshold1_v1', 
    'HLT_Physics_v1', 
    'HLT_PixelTracks_Multiplicity100_v2', 
    'HLT_PixelTracks_Multiplicity80_v2', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_DoubleMu5_Ele8_CaloIdL_TrkIdVL_v3', 
    'HLT_DoubleMu5_Ele8_v3', 
    'HLT_Mu10_Ele10_CaloIdL_v3', 
    'HLT_Mu15_DoublePhoton15_CaloIdL_v3', 
    'HLT_Mu15_Photon20_CaloIdL_v3', 
    'HLT_Mu17_Ele8_CaloIdL_v2', 
    'HLT_Mu5_DoubleEle8_v3', 
    'HLT_Mu5_Ele8_CaloIdL_TrkIdVL_Ele8_v3', 
    'HLT_Mu8_Ele17_CaloIdL_v2', 
    'HLT_Mu8_Photon20_CaloIdVT_IsoT_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleMu3_HT160_v3', 
    'HLT_DoubleMu3_HT200_v3', 
    'HLT_IsoMu17_CentralJet40_BTagIP_v2', 
    'HLT_Mu17_CentralJet30_v2', 
    'HLT_Mu17_CentralJet40_BTagIP_v2', 
    'HLT_Mu17_DiCentralJet30_v2', 
    'HLT_Mu17_TriCentralJet30_v2', 
    'HLT_Mu3_Ele8_CaloIdL_TrkIdVL_HT160_v3', 
    'HLT_Mu3_Ele8_CaloIdT_TrkIdVL_HT160_v3', 
    'HLT_Mu5_HT200_v4', 
    'HLT_Mu8_HT200_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_DoubleMu2_Bs_v1', 
    'HLT_DoubleMu3_Jpsi_v2', 
    'HLT_DoubleMu3_LowMass_v1', 
    'HLT_DoubleMu3_Quarkonium_v2', 
    'HLT_DoubleMu3_Upsilon_v1', 
    'HLT_Mu3_Track3_Jpsi_v5', 
    'HLT_Mu5_L2Mu2_Jpsi_v2', 
    'HLT_Mu5_L2Mu2_v2', 
    'HLT_Mu5_Track2_Jpsi_v1', 
    'HLT_Mu7_Track5_Jpsi_v2', 
    'HLT_Mu7_Track7_Jpsi_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_DoubleJet30_ForwardBackward_v2', 
    'HLT_DoubleJet60_ForwardBackward_v2', 
    'HLT_DoubleJet70_ForwardBackward_v2', 
    'HLT_DoubleJet80_ForwardBackward_v2', 
    'HLT_ExclDiJet60_HFAND_v1', 
    'HLT_ExclDiJet60_HFOR_v1', 
    'HLT_QuadJet40_IsoPFTau40_v1', 
    'HLT_QuadJet40_v2', 
    'HLT_QuadJet50_BTagIP_v1', 
    'HLT_QuadJet50_Jet40_v1', 
    'HLT_QuadJet60_v1', 
    'HLT_QuadJet70_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhoton_selector
streamA_datasetPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPhoton_selector.throw      = cms.bool(False)
streamA_datasetPhoton_selector.triggerConditions = cms.vstring('HLT_DoublePhoton33_v2', 
    'HLT_DoublePhoton5_IsoVL_CEP_v1', 
    'HLT_Photon125_NoSpikeFilter_v2', 
    'HLT_Photon20_CaloIdVL_IsoL_v1', 
    'HLT_Photon20_R9Id_Photon18_R9Id_v2', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_CaloIdL_IsoVL_v2', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_R9Id_v1', 
    'HLT_Photon26_CaloIdL_IsoVL_Photon18_v2', 
    'HLT_Photon26_IsoVL_Photon18_IsoVL_v2', 
    'HLT_Photon26_IsoVL_Photon18_v2', 
    'HLT_Photon26_Photon18_v2', 
    'HLT_Photon26_R9Id_Photon18_CaloIdL_IsoVL_v1', 
    'HLT_Photon30_CaloIdVL_IsoL_v2', 
    'HLT_Photon30_CaloIdVL_v2', 
    'HLT_Photon32_CaloIdL_Photon26_CaloIdL_v2', 
    'HLT_Photon36_CaloIdL_Photon22_CaloIdL_v1', 
    'HLT_Photon50_CaloIdVL_IsoL_v1', 
    'HLT_Photon75_CaloIdVL_IsoL_v2', 
    'HLT_Photon75_CaloIdVL_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon60_CaloIdL_HT200_v2', 
    'HLT_Photon70_CaloIdL_HT200_v2', 
    'HLT_Photon70_CaloIdL_HT300_v2', 
    'HLT_Photon70_CaloIdL_MHT30_v2', 
    'HLT_Photon70_CaloIdL_MHT50_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v2', 
    'HLT_Ele32_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1', 
    'HLT_Ele45_CaloIdVT_TrkIdT_v2', 
    'HLT_Ele90_NoSpikeFilter_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_IsoMu12_v1', 
    'HLT_IsoMu15_v5', 
    'HLT_IsoMu17_v5', 
    'HLT_IsoMu24_v1', 
    'HLT_IsoMu30_v1', 
    'HLT_L1SingleMu10_v1', 
    'HLT_L1SingleMu20_v1', 
    'HLT_L2Mu10_v1', 
    'HLT_L2Mu20_v1', 
    'HLT_Mu12_v1', 
    'HLT_Mu15_v2', 
    'HLT_Mu20_v1', 
    'HLT_Mu24_v1', 
    'HLT_Mu30_v1', 
    'HLT_Mu3_v3', 
    'HLT_Mu5_v3', 
    'HLT_Mu8_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleIsoPFTau20_Trk5_v2', 
    'HLT_IsoPFTau35_Trk20_MET45_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v2', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v2', 
    'HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v2', 
    'HLT_Ele15_CaloIdVT_TrkIdT_LooseIsoPFTau15_v2', 
    'HLT_IsoMu12_LooseIsoPFTau10_v2', 
    'HLT_Mu15_LooseIsoPFTau20_v2')

