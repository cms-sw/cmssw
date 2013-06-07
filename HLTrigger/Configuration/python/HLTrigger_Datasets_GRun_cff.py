# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBJetPlusX_selector
streamA_datasetBJetPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBJetPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetBJetPlusX_selector.throw      = cms.bool(False)
streamA_datasetBJetPlusX_selector.triggerConditions = cms.vstring('HLT_DiJet110Eta2p6_BTagIP3DFastPVLoose_v1', 
    'HLT_DiJet40Eta2p6_BTagIP3DFastPV_v8', 
    'HLT_DiJet80Eta2p6_BTagIP3DFastPVLoose_v8', 
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05_v6', 
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d03_v6', 
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d05_v6', 
    'HLT_DiPFJet95_DiPFJet35_BTagCSVd07d05_v1', 
    'HLT_DiPFJet95_DiPFJet35_BTagCSVd07d05d03_v1', 
    'HLT_DiPFJet95_DiPFJet35_BTagCSVd07d05d05_v1', 
    'HLT_Jet100Eta1p7_Jet85Eta1p7_DiBTagIP3DFastPV_v1', 
    'HLT_Jet160Eta2p4_Jet120Eta2p4_DiBTagIP3DFastPVLoose_v8', 
    'HLT_Jet190Eta2p4_Jet145Eta2p4_DiBTagIP3DFastPVLoose_v1', 
    'HLT_Jet60Eta1p7_Jet53Eta1p7_DiBTagIP3DFastPV_v8', 
    'HLT_Jet80Eta1p7_Jet70Eta1p7_DiBTagIP3DFastPV_v8', 
    'HLT_L1DoubleJet36Central_v7', 
    'HLT_QuadJet75_55_35_20_BTagIP_VBF_v9', 
    'HLT_QuadJet75_55_35_20_VBF_v2', 
    'HLT_QuadJet75_55_38_20_BTagIP_VBF_v9', 
    'HLT_QuadJet90_65_45_25_BTagIP_VBF_v1', 
    'HLT_QuadPFJet78_61_44_31_BTagCSV_VBF_v7', 
    'HLT_QuadPFJet78_61_44_31_VBF_v2', 
    'HLT_QuadPFJet82_65_48_35_BTagCSV_VBF_v7', 
    'HLT_QuadPFJet95_75_55_40_BTagCSV_VBF_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTag_selector
streamA_datasetBTag_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTag_selector.l1tResults = cms.InputTag('')
streamA_datasetBTag_selector.throw      = cms.bool(False)
streamA_datasetBTag_selector.triggerConditions = cms.vstring('HLT_BTagMu_DiJet110_Mu5_v7', 
    'HLT_BTagMu_DiJet20_Mu5_v7', 
    'HLT_BTagMu_DiJet40_Mu5_v7', 
    'HLT_BTagMu_DiJet70_Mu5_v7', 
    'HLT_BTagMu_Jet300_Mu5_v7', 
    'HLT_BTagMu_Jet375_Mu5_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v14', 
    'HLT_BeamGas_HF_Beam1_v5', 
    'HLT_BeamGas_HF_Beam2_v5', 
    'HLT_IsoTrackHB_v15', 
    'HLT_IsoTrackHE_v16', 
    'HLT_L1SingleEG12_v6', 
    'HLT_L1SingleEG5_v6', 
    'HLT_L1SingleJet16_v7', 
    'HLT_L1SingleJet36_v7', 
    'HLT_L1SingleMu12_v2', 
    'HLT_L1SingleMuOpen_v7', 
    'HLT_L1Tech_DT_GlobalOR_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v13', 
    'HLT_L1SingleMuOpen_AntiBPTX_v7', 
    'HLT_L1TrackerCosmics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleElectron_selector
streamA_datasetDoubleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleElectron_selector.throw      = cms.bool(False)
streamA_datasetDoubleElectron_selector.triggerConditions = cms.vstring('HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v13', 
    'HLT_DoubleEle13_CaloIdL_TrkIdVL_Ele13_CaloIdT_TrkIdVL_v1', 
    'HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v7', 
    'HLT_Ele17_CaloIdL_CaloIsoVL_v18', 
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v20', 
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v8', 
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v7', 
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass50_v7', 
    'HLT_Ele18_Ele12_Ele10_CaloIdL_TrkIdVL_v1', 
    'HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v8', 
    'HLT_Ele23_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT30_v9', 
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v9', 
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT15_v9', 
    'HLT_Ele30_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v1', 
    'HLT_Ele30_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT35_v1', 
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_Mass50_v7', 
    'HLT_Ele38_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele20_CaloIdT_CaloIsoVL_trackless_v1', 
    'HLT_Ele38_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT20_v1', 
    'HLT_Ele5_SC5_Jpsi_Mass2to15_v6', 
    'HLT_Ele8_CaloIdL_CaloIsoVL_v18', 
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v8', 
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v16', 
    'HLT_Ele8_CaloIdT_TrkIdVL_EG7_v3', 
    'HLT_Ele8_CaloIdT_TrkIdVL_Jet30_v8', 
    'HLT_Ele8_CaloIdT_TrkIdVL_v6', 
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_v6', 
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_v6', 
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_v6', 
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_v6', 
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_v6', 
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v19', 
    'HLT_TripleEle13_CaloIdL_TrkIdVL_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMu_selector
streamA_datasetDoubleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMu_selector.throw      = cms.bool(False)
streamA_datasetDoubleMu_selector.triggerConditions = cms.vstring('HLT_DoubleMu11_Acoplanarity03_v6', 
    'HLT_DoubleMu20_Acoplanarity03_v1', 
    'HLT_DoubleMu4_Acoplanarity03_v6', 
    'HLT_DoubleMu5_IsoMu5_v21', 
    'HLT_DoubleMu6_IsoMu6_v1', 
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v3', 
    'HLT_L2DoubleMu23_NoVertex_v11', 
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v3', 
    'HLT_L2DoubleMu48_NoVertex_2Cha_Angle2p5_v1', 
    'HLT_Mu13_Mu8_NoDZ_v2', 
    'HLT_Mu17_Mu8_v23', 
    'HLT_Mu17_TkMu8_NoDZ_v2', 
    'HLT_Mu17_TkMu8_v15', 
    'HLT_Mu17_v6', 
    'HLT_Mu22_TkMu22_v10', 
    'HLT_Mu22_TkMu8_v10', 
    'HLT_Mu23_Mu10_v1', 
    'HLT_Mu23_TkMu10_NoDZ_v1', 
    'HLT_Mu23_TkMu10_v1', 
    'HLT_Mu30_TkMu10_v1', 
    'HLT_Mu30_TkMu30_v1', 
    'HLT_Mu8_v19', 
    'HLT_TripleMu5_v20', 
    'HLT_TripleMu6_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMuParked_selector
streamA_datasetDoubleMuParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMuParked_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMuParked_selector.throw      = cms.bool(False)
streamA_datasetDoubleMuParked_selector.triggerConditions = cms.vstring('HLT_DoubleMu11_Acoplanarity03_v6', 
    'HLT_DoubleMu4_Acoplanarity03_v6', 
    'HLT_DoubleMu5_IsoMu5_v21', 
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v3', 
    'HLT_L2DoubleMu23_NoVertex_v11', 
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v3', 
    'HLT_Mu13_Mu8_NoDZ_v2', 
    'HLT_Mu13_Mu8_v23', 
    'HLT_Mu17_Mu8_v23', 
    'HLT_Mu17_TkMu8_NoDZ_v2', 
    'HLT_Mu17_TkMu8_v15', 
    'HLT_Mu17_v6', 
    'HLT_Mu22_TkMu22_v10', 
    'HLT_Mu22_TkMu8_v10', 
    'HLT_Mu8_v19', 
    'HLT_TripleMu5_v20')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoublePhoton_selector
streamA_datasetDoublePhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoublePhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetDoublePhoton_selector.throw      = cms.bool(False)
streamA_datasetDoublePhoton_selector.triggerConditions = cms.vstring('HLT_Photon26_Photon18_v13', 
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass70_v3', 
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_v6', 
    'HLT_Photon34_R9Id85_OR_CaloId10_Iso50_Photon24_R9Id85_OR_CaloId10_Iso50_Mass70_v1', 
    'HLT_Photon36_CaloId10_Iso50_Photon22_CaloId10_Iso50_v7', 
    'HLT_Photon36_CaloId10_Iso50_Photon22_R9Id85_v7', 
    'HLT_Photon36_Photon22_v7', 
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon10_R9Id85_OR_CaloId10_Iso50_Mass80_v2', 
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v7', 
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_v6', 
    'HLT_Photon36_R9Id85_Photon22_CaloId10_Iso50_v7', 
    'HLT_Photon36_R9Id85_Photon22_R9Id85_v5', 
    'HLT_Photon42_CaloId10_Iso50_Photon28_CaloId10_Iso50_v1', 
    'HLT_Photon42_CaloId10_Iso50_Photon28_R9Id85_v1', 
    'HLT_Photon42_R9Id85_OR_CaloId10_Iso50_Photon15_R9Id85_OR_CaloId10_Iso50_Mass80_v1', 
    'HLT_Photon42_R9Id85_OR_CaloId10_Iso50_Photon28_R9Id85_OR_CaloId10_Iso50_v1', 
    'HLT_Photon42_R9Id85_Photon28_CaloId10_Iso50_v1', 
    'HLT_Photon42_R9Id85_Photon28_R9Id85_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoublePhotonHighPt_selector
streamA_datasetDoublePhotonHighPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoublePhotonHighPt_selector.l1tResults = cms.InputTag('')
streamA_datasetDoublePhotonHighPt_selector.throw      = cms.bool(False)
streamA_datasetDoublePhotonHighPt_selector.triggerConditions = cms.vstring('HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v8', 
    'HLT_DoubleEle33_CaloIdL_v15', 
    'HLT_DoubleEle33_CaloIdT_v11', 
    'HLT_DoubleEle40_CaloIdL_GsfTrkIdVL_v1', 
    'HLT_DoubleEle40_CaloIdT_v1', 
    'HLT_DoublePhoton40_CaloIdL_Rsq0p035_v7', 
    'HLT_DoublePhoton40_CaloIdL_Rsq0p06_v7', 
    'HLT_DoublePhoton46_CaloIdL_Rsq0p035_v1', 
    'HLT_DoublePhoton46_CaloIdL_Rsq0p06_v1', 
    'HLT_DoublePhoton48_HEVT_v10', 
    'HLT_DoublePhoton53_HEVT_v4', 
    'HLT_DoublePhoton56_HEVT_v1', 
    'HLT_DoublePhoton61_HEVT_v1', 
    'HLT_DoublePhoton70_v7', 
    'HLT_DoublePhoton80_v8', 
    'HLT_DoublePhoton92_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetElectronHad_selector
streamA_datasetElectronHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetElectronHad_selector.l1tResults = cms.InputTag('')
streamA_datasetElectronHad_selector.throw      = cms.bool(False)
streamA_datasetElectronHad_selector.triggerConditions = cms.vstring('HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v4', 
    'HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v4', 
    'HLT_CleanPFNoPUHT300_Ele40_CaloIdVT_TrkIdT_v4', 
    'HLT_CleanPFNoPUHT300_Ele55_CaloIdVT_TrkIdT_v1', 
    'HLT_CleanPFNoPUHT300_Ele60_CaloIdVT_TrkIdT_v4', 
    'HLT_CleanPFNoPUHT300_Ele85_CaloIdVT_TrkIdT_v1', 
    'HLT_CleanPFNoPUHT350_Ele18_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v1', 
    'HLT_CleanPFNoPUHT350_Ele18_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET60_v1', 
    'HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v4', 
    'HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v4', 
    'HLT_CleanPFNoPUHT400_Ele8_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v1', 
    'HLT_CleanPFNoPUHT400_Ele8_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET60_v1', 
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET40_v9', 
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET50_v9', 
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET55_v1', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v5', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v5', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT325_v1', 
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_v13', 
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_DoubleCentralJet65_v5', 
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR30_Rsq0p04_MR200_v5', 
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR35_Rsq0p04_MR200_v1', 
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR40_Rsq0p04_MR200_v5', 
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR47_Rsq0p04_MR200_v1', 
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet100_PFNoPUJet25_v9', 
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet125_PFNoPUJet25_v1', 
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet150_PFNoPUJet25_v9', 
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet180_PFNoPUJet25_v1', 
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v19', 
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v19', 
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v19')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFEDMonitor_selector
streamA_datasetFEDMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFEDMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetFEDMonitor_selector.throw      = cms.bool(False)
streamA_datasetFEDMonitor_selector.triggerConditions = cms.vstring('HLT_DTErrors_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetForwardTriggers_selector
streamA_datasetForwardTriggers_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetForwardTriggers_selector.l1tResults = cms.InputTag('')
streamA_datasetForwardTriggers_selector.throw      = cms.bool(False)
streamA_datasetForwardTriggers_selector.triggerConditions = cms.vstring('HLT_L1Tech_CASTOR_HaloMuon_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHLTPhysicsParked_selector
streamA_datasetHLTPhysicsParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHLTPhysicsParked_selector.l1tResults = cms.InputTag('')
streamA_datasetHLTPhysicsParked_selector.throw      = cms.bool(False)
streamA_datasetHLTPhysicsParked_selector.triggerConditions = cms.vstring('HLT_Physics_Parked_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHTMHT_selector
streamA_datasetHTMHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHTMHT_selector.l1tResults = cms.InputTag('')
streamA_datasetHTMHT_selector.throw      = cms.bool(False)
streamA_datasetHTMHT_selector.triggerConditions = cms.vstring('HLT_HT250_AlphaT0p55_v8', 
    'HLT_HT250_AlphaT0p57_v8', 
    'HLT_HT285_AlphaT0p55_v1', 
    'HLT_HT290_AlphaT0p57_v1', 
    'HLT_HT300_AlphaT0p53_v8', 
    'HLT_HT300_AlphaT0p54_v14', 
    'HLT_HT340_AlphaT0p53_v1', 
    'HLT_HT340_AlphaT0p54_v1', 
    'HLT_HT350_AlphaT0p52_v8', 
    'HLT_HT350_AlphaT0p53_v19', 
    'HLT_HT380_AlphaT0p53_v1', 
    'HLT_HT385_AlphaT0p52_v1', 
    'HLT_HT400_AlphaT0p51_v19', 
    'HLT_HT400_AlphaT0p52_v14', 
    'HLT_HT440_AlphaT0p51_v1', 
    'HLT_HT445_AlphaT0p52_v1', 
    'HLT_HT450_AlphaT0p51_v14', 
    'HLT_HT500_AlphaT0p51_v1', 
    'HLT_PFNoPUHT350_PFMET100_v5', 
    'HLT_PFNoPUHT400_PFMET100_v5', 
    'HLT_PFNoPUHT405_PFMET100_v1', 
    'HLT_PFNoPUHT450_PFMET100_v1', 
    'HLT_RsqMR40_Rsq0p04_v6', 
    'HLT_RsqMR55_Rsq0p09_MR150_v6', 
    'HLT_RsqMR60_Rsq0p09_MR150_v6', 
    'HLT_RsqMR65_Rsq0p09_MR150_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHTMHTParked_selector
streamA_datasetHTMHTParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHTMHTParked_selector.l1tResults = cms.InputTag('')
streamA_datasetHTMHTParked_selector.throw      = cms.bool(False)
streamA_datasetHTMHTParked_selector.triggerConditions = cms.vstring('HLT_HT200_AlphaT0p57_v8', 
    'HLT_HT250_AlphaT0p55_v8', 
    'HLT_HT250_AlphaT0p57_v8', 
    'HLT_HT300_AlphaT0p53_v8', 
    'HLT_HT300_AlphaT0p54_v14', 
    'HLT_HT350_AlphaT0p52_v8', 
    'HLT_HT350_AlphaT0p53_v19', 
    'HLT_HT400_AlphaT0p51_v19', 
    'HLT_HT400_AlphaT0p52_v14', 
    'HLT_HT450_AlphaT0p51_v14', 
    'HLT_PFNoPUHT350_PFMET100_v5', 
    'HLT_PFNoPUHT400_PFMET100_v5', 
    'HLT_RsqMR40_Rsq0p04_v6', 
    'HLT_RsqMR45_Rsq0p09_v5', 
    'HLT_RsqMR55_Rsq0p09_MR150_v6', 
    'HLT_RsqMR60_Rsq0p09_MR150_v6', 
    'HLT_RsqMR65_Rsq0p09_MR150_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v8', 
    'HLT_L1Tech_HBHEHO_totalOR_v6', 
    'HLT_L1Tech_HCAL_HF_single_channel_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v10', 
    'HLT_HcalPhiSym_v11', 
    'HLT_HcalUTCA_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetHT_selector
streamA_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetJetHT_selector.throw      = cms.bool(False)
streamA_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_DiPFJetAve320_v11', 
    'HLT_DiPFJetAve400_v11', 
    'HLT_DiPFJetAve450_v1', 
    'HLT_FatDiPFJetMass750_DR1p1_Deta1p5_v11', 
    'HLT_FatDiPFJetMass850_DR1p1_Deta1p5_v1', 
    'HLT_HT200_v6', 
    'HLT_HT250_v7', 
    'HLT_HT300_DoubleDisplacedPFJet60_ChgFraction10_v11', 
    'HLT_HT300_DoubleDisplacedPFJet60_v11', 
    'HLT_HT300_SingleDisplacedPFJet60_ChgFraction10_v11', 
    'HLT_HT300_SingleDisplacedPFJet60_v11', 
    'HLT_HT300_v7', 
    'HLT_HT350_v7', 
    'HLT_HT360_DoubleDisplacedPFJet60_v1', 
    'HLT_HT360_SingleDisplacedPFJet60_v1', 
    'HLT_HT400_v7', 
    'HLT_HT450_v7', 
    'HLT_HT500_v7', 
    'HLT_HT550_v7', 
    'HLT_HT650_Track50_dEdx3p6_v11', 
    'HLT_HT650_Track60_dEdx3p7_v11', 
    'HLT_HT650_v7', 
    'HLT_HT750_v7', 
    'HLT_HT820_Track50_dEdx3p6_v1', 
    'HLT_HT820_Track60_dEdx3p7_v1', 
    'HLT_HT850_v1', 
    'HLT_Jet370_NoJetID_v15', 
    'HLT_Jet420_NoJetID_v1', 
    'HLT_MET80_Track50_dEdx3p6_v7', 
    'HLT_MET80_Track60_dEdx3p7_v7', 
    'HLT_MET80_v5', 
    'HLT_MET85_Track50_dEdx3p6_v1', 
    'HLT_MET85_Track60_dEdx3p7_v1', 
    'HLT_PFJet320_v10', 
    'HLT_PFJet360_v1', 
    'HLT_PFJet400_v10', 
    'HLT_PFJet450_v1', 
    'HLT_PFNoPUHT350_v5', 
    'HLT_PFNoPUHT650_DiCentralPFNoPUJet80_CenPFNoPUJet40_v5', 
    'HLT_PFNoPUHT650_v5', 
    'HLT_PFNoPUHT700_v5', 
    'HLT_PFNoPUHT735_DiCentralPFNoPUJet80_CenPFNoPUJet40_v1', 
    'HLT_PFNoPUHT735_v1', 
    'HLT_PFNoPUHT750_v5', 
    'HLT_PFNoPUHT800_v1', 
    'HLT_PFNoPUHT850_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetMon_selector
streamA_datasetJetMon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetMon_selector.l1tResults = cms.InputTag('')
streamA_datasetJetMon_selector.throw      = cms.bool(False)
streamA_datasetJetMon_selector.triggerConditions = cms.vstring('HLT_DiPFJetAve140_v11', 
    'HLT_DiPFJetAve200_v11', 
    'HLT_DiPFJetAve260_v11', 
    'HLT_DiPFJetAve40_v10', 
    'HLT_DiPFJetAve80_v11', 
    'HLT_PFJet140_v10', 
    'HLT_PFJet200_v10', 
    'HLT_PFJet260_v10', 
    'HLT_PFJet40_v9', 
    'HLT_PFJet80_v10', 
    'HLT_SingleForJet15_v4', 
    'HLT_SingleForJet25_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetLogMonitor_selector
streamA_datasetLogMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetLogMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetLogMonitor_selector.throw      = cms.bool(False)
streamA_datasetLogMonitor_selector.triggerConditions = cms.vstring('HLT_LogMonitor_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMET_selector
streamA_datasetMET_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMET_selector.l1tResults = cms.InputTag('')
streamA_datasetMET_selector.throw      = cms.bool(False)
streamA_datasetMET_selector.triggerConditions = cms.vstring('HLT_DiCentralJetSumpT100_dPhi05_DiCentralPFJet60_25_PFMET100_HBHENoiseCleaned_v6', 
    'HLT_DiCentralJetSumpT100_dPhi05_DiCentralPFJet60_25_PFMET120_HBHENoiseCleaned_v1', 
    'HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v6', 
    'HLT_DiCentralPFJet30_PFMET80_v7', 
    'HLT_DiCentralPFJet30_PFMET90_BTagCSV07_v1', 
    'HLT_DiCentralPFNoPUJet50_PFMETORPFMETNoMu100_v1', 
    'HLT_DiCentralPFNoPUJet50_PFMETORPFMETNoMu80_v5', 
    'HLT_DiPFJet40_PFMETnoMu65_MJJ600VBF_LeadingJets_v10', 
    'HLT_DiPFJet40_PFMETnoMu65_MJJ800VBF_AllJets_v10', 
    'HLT_DiPFJet40_PFMETnoMu75_MJJ600VBF_LeadingJets_v1', 
    'HLT_DiPFJet40_PFMETnoMu75_MJJ800VBF_AllJets_v1', 
    'HLT_L1ETM100_v2', 
    'HLT_L1ETM30_v2', 
    'HLT_L1ETM40_v2', 
    'HLT_L1ETM70_v2', 
    'HLT_MET120_HBHENoiseCleaned_v7', 
    'HLT_MET120_v13', 
    'HLT_MET140_HBHENoiseCleaned_v1', 
    'HLT_MET200_HBHENoiseCleaned_v6', 
    'HLT_MET200_v12', 
    'HLT_MET230_HBHENoiseCleaned_v1', 
    'HLT_MET250_v1', 
    'HLT_MET300_HBHENoiseCleaned_v6', 
    'HLT_MET300_v4', 
    'HLT_MET375_v1', 
    'HLT_MET400_HBHENoiseCleaned_v6', 
    'HLT_MET400_v7', 
    'HLT_MET500_HBHENoiseCleaned_v1', 
    'HLT_MET500_v1', 
    'HLT_MonoCentralPFJet150_PFMETnoMu105_NHEF0p95_v1', 
    'HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v5', 
    'HLT_PFMET150_v8', 
    'HLT_PFMET180_v8', 
    'HLT_PFMET200_v1', 
    'HLT_PFMET230_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMETParked_selector
streamA_datasetMETParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMETParked_selector.l1tResults = cms.InputTag('')
streamA_datasetMETParked_selector.throw      = cms.bool(False)
streamA_datasetMETParked_selector.triggerConditions = cms.vstring('HLT_DiCentralJetSumpT100_dPhi05_DiCentralPFJet60_25_PFMET100_HBHENoiseCleaned_v6', 
    'HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v6', 
    'HLT_DiCentralPFJet30_PFMET80_v7', 
    'HLT_DiCentralPFNoPUJet50_PFMETORPFMETNoMu80_v5', 
    'HLT_DiPFJet40_PFMETnoMu65_MJJ600VBF_LeadingJets_v10', 
    'HLT_DiPFJet40_PFMETnoMu65_MJJ800VBF_AllJets_v10', 
    'HLT_L1ETM100_v2', 
    'HLT_L1ETM30_v2', 
    'HLT_L1ETM40_v2', 
    'HLT_L1ETM70_v2', 
    'HLT_MET100_HBHENoiseCleaned_v2', 
    'HLT_MET120_HBHENoiseCleaned_v7', 
    'HLT_MET120_v13', 
    'HLT_MET200_HBHENoiseCleaned_v6', 
    'HLT_MET200_v12', 
    'HLT_MET300_HBHENoiseCleaned_v6', 
    'HLT_MET300_v4', 
    'HLT_MET400_HBHENoiseCleaned_v6', 
    'HLT_MET400_v7', 
    'HLT_MET80_Parked_v5', 
    'HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v5', 
    'HLT_PFMET150_v8', 
    'HLT_PFMET180_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_Physics_v5', 
    'HLT_PixelTracks_Multiplicity70_v4', 
    'HLT_PixelTracks_Multiplicity80_v13', 
    'HLT_PixelTracks_Multiplicity90_v4', 
    'HLT_Random_v2', 
    'HLT_ZeroBiasPixel_DoubleTrack_v2', 
    'HLT_ZeroBias_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuEG_selector
streamA_datasetMuEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuEG_selector.l1tResults = cms.InputTag('')
streamA_datasetMuEG_selector.throw      = cms.bool(False)
streamA_datasetMuEG_selector.triggerConditions = cms.vstring('HLT_DoubleMu10_Ele10_CaloIdT_TrkIdVL_v1', 
    'HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v17', 
    'HLT_DoubleMu8_Ele10_CaloIdT_TrkIdVL_v1', 
    'HLT_DoubleMu8_Ele8_CaloIdT_TrkIdVL_v6', 
    'HLT_Mu10_DoubleEle10_CaloIdT_TrkIdVL_v1', 
    'HLT_Mu10_Ele10_CaloIdT_TrkIdVL_Ele10_CaloIdL_TrkIdVL_v1', 
    'HLT_Mu10_Ele20_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v1', 
    'HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v10', 
    'HLT_Mu20_Ele10_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v1', 
    'HLT_Mu22_Photon22_CaloIdL_v8', 
    'HLT_Mu26_Photon26_CaloIdL_v1', 
    'HLT_Mu30_Ele30_CaloIdL_v9', 
    'HLT_Mu36_Ele36_CaloIdL_v1', 
    'HLT_Mu7_Ele7_CaloIdT_CaloIsoVL_v8', 
    'HLT_Mu8_DoubleEle8_CaloIdT_TrkIdVL_v8', 
    'HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v10', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuHad_selector
streamA_datasetMuHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuHad_selector.l1tResults = cms.InputTag('')
streamA_datasetMuHad_selector.throw      = cms.bool(False)
streamA_datasetMuHad_selector.triggerConditions = cms.vstring('HLT_DoubleDisplacedMu4_DiPFJet40Neutral_v9', 
    'HLT_DoubleDisplacedMu6_DiPFJet40Neutral_v1', 
    'HLT_DoubleMu14_Mass8_PFMET40_v9', 
    'HLT_DoubleMu14_Mass8_PFMET50_v9', 
    'HLT_DoubleMu14_Mass8_PFMET60_v1', 
    'HLT_DoubleMu8_Mass8_PFNoPUHT175_v5', 
    'HLT_DoubleMu8_Mass8_PFNoPUHT225_v5', 
    'HLT_DoubleMu8_Mass8_PFNoPUHT300_v1', 
    'HLT_DoubleMu8_Mass8_PFNoPUHT340_v1', 
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT175_v5', 
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT225_v5', 
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT300_v1', 
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT325_v1', 
    'HLT_IsoMu12_DoubleCentralJet65_v5', 
    'HLT_IsoMu12_RsqMR30_Rsq0p04_MR200_v5', 
    'HLT_IsoMu12_RsqMR40_Rsq0p04_MR200_v5', 
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_PFNoPUHT350_PFMHT40_v4', 
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_PFNoPUHT350_PFMHT60_v1', 
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_PFNoPUHT410_PFMHT40_v1', 
    'HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_v9', 
    'HLT_L2TripleMu17_4_4_NoVertex_PFJet40Neutral_v1', 
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET40_v9', 
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET50_v9', 
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET60_v1', 
    'HLT_Mu40_PFNoPUHT350_v5', 
    'HLT_Mu40_PFNoPUHT410_v1', 
    'HLT_Mu60_PFNoPUHT350_v5', 
    'HLT_Mu60_PFNoPUHT470_v1', 
    'HLT_Mu8_DiJet30_v8', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v6', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v6', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT300_v1', 
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT325_v1', 
    'HLT_Mu8_QuadJet30_v8', 
    'HLT_Mu8_QuadJet42_v1', 
    'HLT_Mu8_TriJet30_v8', 
    'HLT_PFNoPUHT350_Mu15_PFMET45_v5', 
    'HLT_PFNoPUHT350_Mu15_PFMET50_v5', 
    'HLT_PFNoPUHT400_Mu15_PFMET50_v1', 
    'HLT_PFNoPUHT400_Mu15_PFMET55_v1', 
    'HLT_PFNoPUHT400_Mu5_PFMET45_v5', 
    'HLT_PFNoPUHT400_Mu5_PFMET50_v5', 
    'HLT_PFNoPUHT450_Mu5_PFMET50_v1', 
    'HLT_PFNoPUHT450_Mu5_PFMET55_v1', 
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v6', 
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v6', 
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT300_v1', 
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT325_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOnia_selector
streamA_datasetMuOnia_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOnia_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOnia_selector.throw      = cms.bool(False)
streamA_datasetMuOnia_selector.triggerConditions = cms.vstring('HLT_Dimuon0_Jpsi_Muon_v19', 
    'HLT_Dimuon0_Jpsi_NoVertexing_v15', 
    'HLT_Dimuon0_Jpsi_v18', 
    'HLT_Dimuon0_PsiPrime_v7', 
    'HLT_Dimuon0_Upsilon_Muon_v19', 
    'HLT_Dimuon0_Upsilon_v18', 
    'HLT_Dimuon11_Upsilon_v7', 
    'HLT_Dimuon3p5_SameSign_v7', 
    'HLT_Dimuon7_Upsilon_v8', 
    'HLT_DoubleMu3_4_Dimuon5_Bs_Central_v6', 
    'HLT_DoubleMu3p5_4_Dimuon5_Bs_Central_v6', 
    'HLT_DoubleMu4_Dimuon7_Bs_Forward_v6', 
    'HLT_DoubleMu4_JpsiTk_Displaced_v8', 
    'HLT_DoubleMu4_Jpsi_Displaced_v13', 
    'HLT_Mu5_L2Mu3_Jpsi_v9', 
    'HLT_Mu5_Track2_Jpsi_v22', 
    'HLT_Mu5_Track3p5_Jpsi_v8', 
    'HLT_Mu7_Track7_Jpsi_v21', 
    'HLT_Tau2Mu_ItTrack_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuOniaParked_selector
streamA_datasetMuOniaParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuOniaParked_selector.l1tResults = cms.InputTag('')
streamA_datasetMuOniaParked_selector.throw      = cms.bool(False)
streamA_datasetMuOniaParked_selector.triggerConditions = cms.vstring('HLT_BTagMu_Jet20_Mu4_v3', 
    'HLT_BTagMu_Jet60_Mu4_v3', 
    'HLT_Dimuon10_Jpsi_v7', 
    'HLT_Dimuon5_PsiPrime_v7', 
    'HLT_Dimuon5_Upsilon_v7', 
    'HLT_Dimuon7_PsiPrime_v4', 
    'HLT_Dimuon8_Jpsi_v8', 
    'HLT_Dimuon8_Upsilon_v7', 
    'HLT_DoubleMu3p5_LowMassNonResonant_Displaced_v7', 
    'HLT_DoubleMu3p5_LowMass_Displaced_v7', 
    'HLT_Mu15_TkMu5_Onia_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet_selector
streamA_datasetMultiJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet_selector.throw      = cms.bool(False)
streamA_datasetMultiJet_selector.triggerConditions = cms.vstring('HLT_DiJet80_DiJet60_DiJet20_v6', 
    'HLT_DiJet80_DiJet60_DiJet30_v1', 
    'HLT_DoubleJet20_ForwardBackward_v4', 
    'HLT_EightJet30_eta3p0_v5', 
    'HLT_EightJet35_eta3p0_v5', 
    'HLT_EightJet40_eta3p0_v1', 
    'HLT_ExclDiJet35_HFAND_v4', 
    'HLT_ExclDiJet35_HFOR_v4', 
    'HLT_ExclDiJet80_HFAND_v4', 
    'HLT_QuadJet100_v1', 
    'HLT_QuadJet60_DiJet20_v6', 
    'HLT_QuadJet60_DiJet30_v1', 
    'HLT_QuadJet70_v6', 
    'HLT_QuadJet80_v6', 
    'HLT_QuadJet90_v6', 
    'HLT_SixJet35_v6', 
    'HLT_SixJet45_v6', 
    'HLT_SixJet50_v6', 
    'HLT_SixJet55_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMultiJet1Parked_selector
streamA_datasetMultiJet1Parked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMultiJet1Parked_selector.l1tResults = cms.InputTag('')
streamA_datasetMultiJet1Parked_selector.throw      = cms.bool(False)
streamA_datasetMultiJet1Parked_selector.triggerConditions = cms.vstring('HLT_DiJet80_DiJet60_DiJet20_v6', 
    'HLT_DoubleJet20_ForwardBackward_v4', 
    'HLT_EightJet30_eta3p0_v5', 
    'HLT_EightJet35_eta3p0_v5', 
    'HLT_ExclDiJet35_HFAND_v4', 
    'HLT_ExclDiJet35_HFOR_v4', 
    'HLT_ExclDiJet80_HFAND_v4', 
    'HLT_QuadJet45_v1', 
    'HLT_QuadJet50_v5', 
    'HLT_QuadJet60_DiJet20_v6', 
    'HLT_QuadJet70_v6', 
    'HLT_QuadJet80_v6', 
    'HLT_QuadJet90_v6', 
    'HLT_SixJet35_v6', 
    'HLT_SixJet45_v6', 
    'HLT_SixJet50_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetNoBPTX_selector
streamA_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamA_datasetNoBPTX_selector.throw      = cms.bool(False)
streamA_datasetNoBPTX_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v16', 
    'HLT_JetE30_NoBPTX_v14', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v13', 
    'HLT_JetE70_NoBPTX3BX_NoHalo_v5', 
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v4', 
    'HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1', 
    'HLT_L2Mu20_eta2p1_NoVertex_v2', 
    'HLT_L2Mu30_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPPMuon_selector
streamA_datasetPPMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPPMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetPPMuon_selector.throw      = cms.bool(False)
streamA_datasetPPMuon_selector.triggerConditions = cms.vstring('HLT_Mu15_eta2p1_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPPPhoton_selector
streamA_datasetPPPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPPPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPPPhoton_selector.throw      = cms.bool(False)
streamA_datasetPPPhoton_selector.triggerConditions = cms.vstring('HLT_Ele22_CaloIdL_CaloIsoVL_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPhotonHad_selector
streamA_datasetPhotonHad_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPhotonHad_selector.l1tResults = cms.InputTag('')
streamA_datasetPhotonHad_selector.throw      = cms.bool(False)
streamA_datasetPhotonHad_selector.triggerConditions = cms.vstring('HLT_Photon110_CaloIdXL_PFMET100_v1', 
    'HLT_Photon40_CaloIdL_RsqMR40_Rsq0p09_MR150_v7', 
    'HLT_Photon40_CaloIdL_RsqMR45_Rsq0p09_MR150_v7', 
    'HLT_Photon40_CaloIdL_RsqMR50_Rsq0p09_MR150_v7', 
    'HLT_Photon40_CaloIdL_RsqMR55_Rsq0p09_MR150_v1', 
    'HLT_Photon60_CaloIdL_HT300_v5', 
    'HLT_Photon60_CaloIdL_MHT70_v12', 
    'HLT_Photon70_CaloIdXL_PFMET100_v8', 
    'HLT_Photon70_CaloIdXL_PFNoPUHT400_v5', 
    'HLT_Photon70_CaloIdXL_PFNoPUHT470_v1', 
    'HLT_Photon70_CaloIdXL_PFNoPUHT500_v5', 
    'HLT_Photon70_CaloIdXL_PFNoPUHT580_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele100_CaloIdVT_GsfTrkIdT_v1', 
    'HLT_Ele115_CaloIdVT_GsfTrkIdT_v1', 
    'HLT_Ele22_CaloIdL_CaloIsoVL_v7', 
    'HLT_Ele24_WP80_CentralPFJet35_CentralPFJet25_PFMET20_v3', 
    'HLT_Ele24_WP80_CentralPFJet35_CentralPFJet25_v3', 
    'HLT_Ele24_WP80_PFJet30_PFJet25_Deta3_CentralPFJet30_v3', 
    'HLT_Ele24_WP80_PFJet30_PFJet25_Deta3_v3', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_BTagIPIter_v11', 
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_v10', 
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_DiCentralPFNoPUJet30_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet30_v6', 
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet45_35_25_v4', 
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet50_40_30_v6', 
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v12', 
    'HLT_Ele27_WP80_CentralPFJet80_v11', 
    'HLT_Ele27_WP80_PFMET_MT50_v9', 
    'HLT_Ele27_WP80_WCandPt80_v11', 
    'HLT_Ele27_WP80_v13', 
    'HLT_Ele28_CaloIdVT_CaloIsoT_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet40_v1', 
    'HLT_Ele28_CaloIdVT_CaloIsoT_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet60_40_25_v1', 
    'HLT_Ele28_CaloIdVT_CaloIsoT_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet65_45_30_v1', 
    'HLT_Ele30_CaloIdVT_TrkIdT_v7', 
    'HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v12', 
    'HLT_Ele36_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet50_BTagIPIter_v1', 
    'HLT_Ele36_WP80_CentralPFJet40_CentralPFJet30_PFMET25_v1', 
    'HLT_Ele36_WP80_CentralPFJet40_CentralPFJet30_v1', 
    'HLT_Ele36_WP80_CentralPFJet95_v1', 
    'HLT_Ele36_WP80_PFJet35_PFJet30_Deta3_CentralPFJet35_v1', 
    'HLT_Ele36_WP80_PFJet35_PFJet30_Deta3_v1', 
    'HLT_Ele36_WP80_PFMET_MT60_v1', 
    'HLT_Ele36_WP80_WCandPt108_v1', 
    'HLT_Ele36_WP80_v1', 
    'HLT_Ele80_CaloIdVT_GsfTrkIdT_v3', 
    'HLT_Ele90_CaloIdVT_GsfTrkIdT_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v5', 
    'HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_v5', 
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_v5', 
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet30_v5', 
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet45_35_25_v3', 
    'HLT_IsoMu18_CentralPFJet30_CentralPFJet25_PFMET20_v2', 
    'HLT_IsoMu18_CentralPFJet30_CentralPFJet25_v2', 
    'HLT_IsoMu18_PFJet30_PFJet25_Deta3_CentralPFJet25_v2', 
    'HLT_IsoMu18_PFJet30_PFJet25_Deta3_v2', 
    'HLT_IsoMu20_WCandPt80_v5', 
    'HLT_IsoMu20_eta2p1_CentralPFJet80_v10', 
    'HLT_IsoMu20_eta2p1_CentralPFNoPUJet40_BTagIPIter_v1', 
    'HLT_IsoMu20_eta2p1_TriCentralPFNoPUJet35_v1', 
    'HLT_IsoMu20_eta2p1_TriCentralPFNoPUJet60_40_25_v1', 
    'HLT_IsoMu20_eta2p1_v8', 
    'HLT_IsoMu22_CentralPFJet35_CentralPFJet30_PFMET22_v1', 
    'HLT_IsoMu22_CentralPFJet35_CentralPFJet30_v1', 
    'HLT_IsoMu22_PFJet33_PFJet28_Deta3_CentralPFJet25_v1', 
    'HLT_IsoMu22_PFJet33_PFJet28_Deta3_v1', 
    'HLT_IsoMu24_eta2p1_v16', 
    'HLT_IsoMu24_v18', 
    'HLT_IsoMu25_WCandPt90_v1', 
    'HLT_IsoMu25_eta2p1_CentralPFJet90_v1', 
    'HLT_IsoMu30_eta2p1_v16', 
    'HLT_IsoMu30_v12', 
    'HLT_IsoMu34_eta2p1_v14', 
    'HLT_IsoMu34_v1', 
    'HLT_IsoMu38_eta2p1_v1', 
    'HLT_IsoMu38_v1', 
    'HLT_IsoMu40_eta2p1_v11', 
    'HLT_IsoMu47_eta2p1_v1', 
    'HLT_L2Mu70_2Cha_eta2p1_PFMET55_v3', 
    'HLT_L2Mu70_2Cha_eta2p1_PFMET60_v3', 
    'HLT_L2Mu75_2Cha_eta2p1_PFMET60_v1', 
    'HLT_L2Mu75_2Cha_eta2p1_PFMET65_v1', 
    'HLT_Mu12_eta2p1_DiCentral_20_v9', 
    'HLT_Mu12_eta2p1_DiCentral_40_20_BTagIP3D1stTrack_v9', 
    'HLT_Mu12_eta2p1_DiCentral_40_20_DiBTagIP3D1stTrack_v9', 
    'HLT_Mu12_eta2p1_DiCentral_40_20_v9', 
    'HLT_Mu12_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v8', 
    'HLT_Mu12_v19', 
    'HLT_Mu15_eta2p1_DiCentral_20_v2', 
    'HLT_Mu15_eta2p1_DiCentral_40_20_v2', 
    'HLT_Mu15_eta2p1_DiCentral_50_20_DiBTagIP3D1stTrack_v1', 
    'HLT_Mu15_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v4', 
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_BTagIP3D1stTrack_v9', 
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_DiBTagIP3D1stTrack_v9', 
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_v9', 
    'HLT_Mu15_eta2p1_v6', 
    'HLT_Mu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v5', 
    'HLT_Mu17_eta2p1_TriCentralPFNoPUJet45_35_25_v3', 
    'HLT_Mu18_CentralPFJet30_CentralPFJet25_v2', 
    'HLT_Mu18_PFJet30_PFJet25_Deta3_CentralPFJet25_v2', 
    'HLT_Mu18_eta2p1_TriCentral_50_20_20_DiBTagIP3D1stTrack_v1', 
    'HLT_Mu24_eta2p1_v6', 
    'HLT_Mu24_v17', 
    'HLT_Mu30_eta2p1_v6', 
    'HLT_Mu30_v17', 
    'HLT_Mu40_eta2p1_Track50_dEdx3p6_v6', 
    'HLT_Mu40_eta2p1_Track60_dEdx3p6_v1', 
    'HLT_Mu40_eta2p1_Track60_dEdx3p7_v6', 
    'HLT_Mu40_eta2p1_Track70_dEdx3p7_v1', 
    'HLT_Mu40_eta2p1_v12', 
    'HLT_Mu40_v15', 
    'HLT_Mu47_eta2p1_v1', 
    'HLT_Mu47_v1', 
    'HLT_Mu50_eta2p1_v9', 
    'HLT_Mu5_v21', 
    'HLT_Mu60_eta2p1_v1', 
    'HLT_RelIso1p0Mu20_v4', 
    'HLT_RelIso1p0Mu5_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSinglePhoton_selector
streamA_datasetSinglePhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSinglePhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetSinglePhoton_selector.throw      = cms.bool(False)
streamA_datasetSinglePhoton_selector.triggerConditions = cms.vstring('HLT_DisplacedPhoton65EBOnly_CaloIdVL_IsoL_PFMET30_v5', 
    'HLT_DisplacedPhoton65_CaloIdVL_IsoL_PFMET25_v5', 
    'HLT_DisplacedPhoton70EBOnly_CaloIdVL_IsoL_PFMET35_v1', 
    'HLT_DisplacedPhoton70_CaloIdVL_IsoL_PFMET30_v1', 
    'HLT_L1DoubleEG3_FwdVeto_v2', 
    'HLT_Photon135_v8', 
    'HLT_Photon150_v5', 
    'HLT_Photon155_v1', 
    'HLT_Photon160_v5', 
    'HLT_Photon175_v1', 
    'HLT_Photon185_v1', 
    'HLT_Photon20_CaloIdVL_IsoL_v17', 
    'HLT_Photon20_CaloIdVL_v5', 
    'HLT_Photon300_NoHE_v6', 
    'HLT_Photon30_CaloIdVL_v15', 
    'HLT_Photon330_NoHE_v1', 
    'HLT_Photon50_CaloIdVL_IsoL_v18', 
    'HLT_Photon50_CaloIdVL_v11', 
    'HLT_Photon75_CaloIdVL_v14', 
    'HLT_Photon90_CaloIdVL_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSinglePhotonParked_selector
streamA_datasetSinglePhotonParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSinglePhotonParked_selector.l1tResults = cms.InputTag('')
streamA_datasetSinglePhotonParked_selector.throw      = cms.bool(False)
streamA_datasetSinglePhotonParked_selector.triggerConditions = cms.vstring('HLT_DisplacedPhoton65EBOnly_CaloIdVL_IsoL_PFMET30_v5', 
    'HLT_DisplacedPhoton65_CaloIdVL_IsoL_PFMET25_v5', 
    'HLT_L1DoubleEG3_FwdVeto_v2', 
    'HLT_Photon135_v8', 
    'HLT_Photon150_v5', 
    'HLT_Photon160_v5', 
    'HLT_Photon20_CaloIdVL_IsoL_v17', 
    'HLT_Photon20_CaloIdVL_v5', 
    'HLT_Photon300_NoHE_v6', 
    'HLT_Photon30_CaloIdVL_v15', 
    'HLT_Photon30_R9Id90_CaloId_HE10_Iso40_EBOnly_Met25_HBHENoiseCleaned_v2', 
    'HLT_Photon30_R9Id90_CaloId_HE10_Iso40_EBOnly_v2', 
    'HLT_Photon30_v2', 
    'HLT_Photon50_CaloIdVL_IsoL_v18', 
    'HLT_Photon50_CaloIdVL_v11', 
    'HLT_Photon75_CaloIdVL_v14', 
    'HLT_Photon90_CaloIdVL_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTau_selector
streamA_datasetTau_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTau_selector.l1tResults = cms.InputTag('')
streamA_datasetTau_selector.throw      = cms.bool(False)
streamA_datasetTau_selector.triggerConditions = cms.vstring('HLT_DoubleIsoL2Tau30_eta2p1_v1', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v6', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Reg_Jet30_v2', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Reg_v2', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v5', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_Reg_v2', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v5', 
    'HLT_DoubleMediumIsoPFTau45_Trk1_eta2p1_Reg_Jet30_v1', 
    'HLT_DoubleMediumIsoPFTau50_Trk1_eta2p1_Prong1_Reg_v1', 
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v11', 
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v11', 
    'HLT_LooseIsoPFTau35_Trk20_Prong1_v11', 
    'HLT_LooseIsoPFTau50_Trk20_Prong1_MET75_v1', 
    'HLT_LooseIsoPFTau50_Trk20_Prong1_MET80_v1', 
    'HLT_LooseIsoPFTau50_Trk20_Prong1_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauParked_selector
streamA_datasetTauParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauParked_selector.l1tResults = cms.InputTag('')
streamA_datasetTauParked_selector.throw      = cms.bool(False)
streamA_datasetTauParked_selector.triggerConditions = cms.vstring('HLT_DoubleIsoL2Tau30_eta2p1_v1', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v6', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Reg_Jet30_v2', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Reg_v2', 
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v5', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_Reg_v2', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v5', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v2', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v5', 
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v11', 
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v11', 
    'HLT_LooseIsoPFTau35_Trk20_Prong1_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTauPlusX_selector
streamA_datasetTauPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTauPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetTauPlusX_selector.throw      = cms.bool(False)
streamA_datasetTauPlusX_selector.triggerConditions = cms.vstring('HLT_Ele13_eta2p1_WP90NoIso_LooseIsoPFTau20_L1ETM36_v2', 
    'HLT_Ele13_eta2p1_WP90Rho_LooseIsoPFTau20_L1ETM36_v2', 
    'HLT_Ele13_eta2p1_WP90Rho_LooseIsoPFTau20_v2', 
    'HLT_Ele22_eta2p1_WP90NoIso_LooseIsoPFTau20_v8', 
    'HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v8', 
    'HLT_Ele25_eta2p1_WP90Rho_LooseIsoPFTau45_L1ETM36_v1', 
    'HLT_Ele25_eta2p1_WP90Rho_LooseIsoPFTau45_v1', 
    'HLT_Ele30_eta2p1_WP90Rho_LooseIsoPFTau45_v1', 
    'HLT_IsoMu15_eta2p1_L1ETM20_v8', 
    'HLT_IsoMu15_eta2p1_LooseIsoPFTau35_Trk20_Prong1_L1ETM20_v11', 
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v8', 
    'HLT_IsoMu18_eta2p1_MediumIsoPFTau25_Trk1_eta2p1_Reg_v2', 
    'HLT_IsoMu18_eta2p1_MediumIsoPFTau25_Trk1_eta2p1_v5', 
    'HLT_IsoMu25_eta2p1_L1ETM20_v1', 
    'HLT_IsoMu25_eta2p1_LooseIsoPFTau35_Trk45_Prong1_L1ETM20_v1', 
    'HLT_IsoMu26_eta2p1_MediumIsoPFTau30_Trk1_eta2p1_Reg_v1', 
    'HLT_IsoMu30_eta2p1_LooseIsoPFTau45_v1', 
    'HLT_IsoMu8_eta2p1_LooseIsoPFTau20_L1ETM26_v2', 
    'HLT_IsoMu8_eta2p1_LooseIsoPFTau20_v2', 
    'HLT_IsoMu8_eta2p1_LooseIsoPFTau40_L1ETM26_v1', 
    'HLT_IsoMu8_eta2p1_LooseIsoPFTau40_v1', 
    'HLT_Mu15_eta2p1_L1ETM20_v6', 
    'HLT_Mu17_eta2p1_LooseIsoPFTau20_v8', 
    'HLT_Mu8_eta2p1_LooseIsoPFTau20_L1ETM26_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetVBF1Parked_selector
streamA_datasetVBF1Parked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetVBF1Parked_selector.l1tResults = cms.InputTag('')
streamA_datasetVBF1Parked_selector.throw      = cms.bool(False)
streamA_datasetVBF1Parked_selector.triggerConditions = cms.vstring('HLT_DiJet20_MJJ650_AllJets_DEta3p5_HT120_VBF_v1', 
    'HLT_DiJet30_MJJ700_AllJets_DEta3p5_VBF_v1', 
    'HLT_DiJet35_MJJ650_AllJets_DEta3p5_VBF_v5', 
    'HLT_DiJet35_MJJ700_AllJets_DEta3p5_VBF_v5', 
    'HLT_DiJet35_MJJ750_AllJets_DEta3p5_VBF_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetZeroBiasParked_selector
streamA_datasetZeroBiasParked_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetZeroBiasParked_selector.l1tResults = cms.InputTag('')
streamA_datasetZeroBiasParked_selector.throw      = cms.bool(False)
streamA_datasetZeroBiasParked_selector.triggerConditions = cms.vstring('HLT_ZeroBias_Parked_v1')

