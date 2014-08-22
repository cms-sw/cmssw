# /online/collisions/2012/7e33/v4.1/HLT/V1 (CMSSW_5_2_6_HLT2)

import FWCore.ParameterSet.Config as cms


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/online/collisions/2012/7e33/v4.1/HLT/V1')
)

streams = cms.PSet( 
  A = cms.vstring( 'BJetPlusX',
    'BTag',
    'Commissioning',
    'Cosmics',
    'DoubleElectron',
    'DoubleMu',
    'DoubleMuParked',
    'DoublePhoton',
    'DoublePhotonHighPt',
    'ElectronHad',
    'FEDMonitor',
    'HTMHT',
    'HTMHTParked',
    'HcalHPDNoise',
    'HcalNZS',
    'JetHT',
    'JetMon',
    'LogMonitor',
    'MET',
    'MinimumBias',
    'MuEG',
    'MuHad',
    'MuOnia',
    'MuOniaParked',
    'MultiJet',
    'MultiJet1Parked',
    'NoBPTX',
    'PhotonHad',
    'SingleElectron',
    'SingleMu',
    'SinglePhoton',
    'Tau',
    'TauParked',
    'TauPlusX',
    'VBF1Parked' ),
  ALCALUMIPIXELS = cms.vstring( 'AlCaLumiPixels' ),
  ALCAP0 = cms.vstring( 'AlCaP0' ),
  ALCAPHISYM = cms.vstring( 'AlCaPhiSym' ),
  B = cms.vstring( 'ParkingMonitor' ),
  Calibration = cms.vstring( 'TestEnablesEcalHcalDT' ),
  DQM = cms.vstring( 'OnlineMonitor' ),
  EcalCalibration = cms.vstring( 'EcalLaser' ),
  Express = cms.vstring( 'ExpressPhysics' ),
  HLTDQM = cms.vstring( 'OnlineHltMonitor' ),
  HLTMON = cms.vstring( 'OfflineMonitor' ),
  NanoDST = cms.vstring( 'L1Accept' ),
  PhysicsDST = cms.vstring( 'DataScouting' ),
  RPCMON = cms.vstring( 'RPCMonitor' ),
  TrackerCalibration = cms.vstring( 'TestEnablesTracker' )
)
datasets = cms.PSet( 
  AlCaLumiPixels = cms.vstring( 'AlCa_LumiPixels_Random_v1',
    'AlCa_LumiPixels_ZeroBias_v4',
    'AlCa_LumiPixels_v8' ),
  AlCaP0 = cms.vstring( 'AlCa_EcalEtaEBonly_v6',
    'AlCa_EcalEtaEEonly_v6',
    'AlCa_EcalPi0EBonly_v6',
    'AlCa_EcalPi0EEonly_v6' ),
  AlCaPhiSym = cms.vstring( 'AlCa_EcalPhiSym_v13' ),
  BJetPlusX = cms.vstring( 'HLT_DiJet40Eta2p6_BTagIP3DFastPV_v7',
    'HLT_DiJet80Eta2p6_BTagIP3DFastPVLoose_v7',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05_v4',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d03_v4',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d05_v4',
    'HLT_Jet160Eta2p4_Jet120Eta2p4_DiBTagIP3DFastPVLoose_v7',
    'HLT_Jet60Eta1p7_Jet53Eta1p7_DiBTagIP3DFastPV_v7',
    'HLT_Jet80Eta1p7_Jet70Eta1p7_DiBTagIP3DFastPV_v7',
    'HLT_L1DoubleJet36Central_v7',
    'HLT_QuadJet75_55_35_20_BTagIP_VBF_v6',
    'HLT_QuadJet75_55_38_20_BTagIP_VBF_v6',
    'HLT_QuadPFJet78_61_44_31_BTagCSV_VBF_v4',
    'HLT_QuadPFJet82_65_48_35_BTagCSV_VBF_v4' ),
  BTag = cms.vstring( 'HLT_BTagMu_DiJet110_Mu5_v6',
    'HLT_BTagMu_DiJet20_Mu5_v6',
    'HLT_BTagMu_DiJet40_Mu5_v6',
    'HLT_BTagMu_DiJet70_Mu5_v6',
    'HLT_BTagMu_Jet300_Mu5_v6' ),
  Commissioning = cms.vstring( 'HLT_Activity_Ecal_SC7_v13',
    'HLT_BeamGas_HF_Beam1_v5',
    'HLT_BeamGas_HF_Beam2_v5',
    'HLT_IsoTrackHB_v14',
    'HLT_IsoTrackHE_v15',
    'HLT_L1SingleEG12_v6',
    'HLT_L1SingleEG5_v6',
    'HLT_L1SingleJet16_v7',
    'HLT_L1SingleJet36_v7',
    'HLT_L1SingleMu12_v2',
    'HLT_L1SingleMuOpen_v7' ),
  Cosmics = cms.vstring( 'HLT_BeamHalo_v13',
    'HLT_L1SingleMuOpen_AntiBPTX_v7',
    'HLT_L1TrackerCosmics_v7' ),
  DataScouting = cms.vstring( 'DST_Ele8_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT250_v4',
    'DST_HT250_v4',
    'DST_L1HTT_Or_L1MultiJet_v4',
    'DST_Mu5_HT250_v4' ),
  DoubleElectron = cms.vstring( 'HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v12',
    'HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v6',
    'HLT_Ele17_CaloIdL_CaloIsoVL_v17',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v18',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v6',
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass50_v6',
    'HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v6',
    'HLT_Ele23_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT30_v7',
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v7',
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT15_v7',
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_Mass50_v6',
    'HLT_Ele5_SC5_Jpsi_Mass2to15_v4',
    'HLT_Ele8_CaloIdL_CaloIsoVL_v17',
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7',
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v15',
    'HLT_Ele8_CaloIdT_TrkIdVL_EG7_v2',
    'HLT_Ele8_CaloIdT_TrkIdVL_Jet30_v7',
    'HLT_Ele8_CaloIdT_TrkIdVL_v5',
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v18' ),
  DoubleMu = cms.vstring( 'HLT_DoubleMu11_Acoplanarity03_v5',
    'HLT_DoubleMu4_Acoplanarity03_v5',
    'HLT_DoubleMu5_IsoMu5_v20',
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v3',
    'HLT_L2DoubleMu23_NoVertex_v11',
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v3',
    'HLT_Mu17_Mu8_v21',
    'HLT_Mu17_TkMu8_v13',
    'HLT_Mu17_v5',
    'HLT_Mu22_TkMu22_v8',
    'HLT_Mu22_TkMu8_v8',
    'HLT_Mu8_v18',
    'HLT_TripleMu5_v19' ),
  DoubleMuParked = cms.vstring( 'HLT_DoubleMu11_Acoplanarity03_v5',
    'HLT_DoubleMu4_Acoplanarity03_v5',
    'HLT_DoubleMu5_IsoMu5_v20',
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v3',
    'HLT_L2DoubleMu23_NoVertex_v11',
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v3',
    'HLT_Mu13_Mu8_v21',
    'HLT_Mu17_Mu8_v21',
    'HLT_Mu17_TkMu8_v13',
    'HLT_Mu17_v5',
    'HLT_Mu22_TkMu22_v8',
    'HLT_Mu22_TkMu8_v8',
    'HLT_Mu8_v18',
    'HLT_TripleMu5_v19' ),
  DoublePhoton = cms.vstring( 'HLT_Photon26_CaloId10_Iso50_Photon18_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_CaloId10_Iso50_Photon18_R9Id85_Mass60_v6',
    'HLT_Photon26_Photon18_v12',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass70_v2',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_v5',
    'HLT_Photon26_R9Id85_Photon18_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_R9Id85_Photon18_R9Id85_Mass60_v4',
    'HLT_Photon36_CaloId10_Iso50_Photon22_CaloId10_Iso50_v6',
    'HLT_Photon36_CaloId10_Iso50_Photon22_R9Id85_v6',
    'HLT_Photon36_Photon22_v6',
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v6',
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_v5',
    'HLT_Photon36_R9Id85_Photon22_CaloId10_Iso50_v6',
    'HLT_Photon36_R9Id85_Photon22_R9Id85_v4' ),
  DoublePhotonHighPt = cms.vstring( 'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v7',
    'HLT_DoubleEle33_CaloIdL_v14',
    'HLT_DoubleEle33_CaloIdT_v10',
    'HLT_DoublePhoton40_CaloIdL_Rsq0p035_v6',
    'HLT_DoublePhoton40_CaloIdL_Rsq0p06_v6',
    'HLT_DoublePhoton48_HEVT_v8',
    'HLT_DoublePhoton53_HEVT_v2',
    'HLT_DoublePhoton70_v6',
    'HLT_DoublePhoton80_v7' ),
  EcalLaser = cms.vstring( 'HLT_EcalCalibration_v3' ),
  ElectronHad = cms.vstring( 'HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v2',
    'HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v2',
    'HLT_CleanPFNoPUHT300_Ele40_CaloIdVT_TrkIdT_v2',
    'HLT_CleanPFNoPUHT300_Ele60_CaloIdVT_TrkIdT_v2',
    'HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v2',
    'HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v2',
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7',
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7',
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3',
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3',
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_v12',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_DoubleCentralJet65_v4',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR30_Rsq0p04_MR200_v4',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR40_Rsq0p04_MR200_v4',
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet100_PFNoPUJet25_v7',
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet150_PFNoPUJet25_v7',
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v18' ),
  ExpressPhysics = cms.vstring( 'HLT_DoublePhoton80_v7',
    'HLT_EightJet30_eta3p0_v5',
    'HLT_EightJet35_eta3p0_v5',
    'HLT_MET400_v7',
    'HLT_Mu17_Mu8_v21',
    'HLT_Photon300_NoHE_v5',
    'HLT_ZeroBias_v7' ),
  FEDMonitor = cms.vstring( 'HLT_DTErrors_v3' ),
  HTMHT = cms.vstring( 'HLT_HT250_AlphaT0p55_v8',
    'HLT_HT250_AlphaT0p57_v8',
    'HLT_HT300_AlphaT0p53_v8',
    'HLT_HT300_AlphaT0p54_v14',
    'HLT_HT350_AlphaT0p52_v8',
    'HLT_HT350_AlphaT0p53_v19',
    'HLT_HT400_AlphaT0p51_v19',
    'HLT_HT400_AlphaT0p52_v14',
    'HLT_HT450_AlphaT0p51_v14',
    'HLT_PFNoPUHT350_PFMET100_v3',
    'HLT_PFNoPUHT400_PFMET100_v3',
    'HLT_RsqMR40_Rsq0p04_v6',
    'HLT_RsqMR55_Rsq0p09_MR150_v6',
    'HLT_RsqMR60_Rsq0p09_MR150_v6',
    'HLT_RsqMR65_Rsq0p09_MR150_v5' ),
  HTMHTParked = cms.vstring( 'HLT_HT200_AlphaT0p57_v8',
    'HLT_HT250_AlphaT0p55_v8',
    'HLT_HT250_AlphaT0p57_v8',
    'HLT_HT300_AlphaT0p53_v8',
    'HLT_HT300_AlphaT0p54_v14',
    'HLT_HT350_AlphaT0p52_v8',
    'HLT_HT350_AlphaT0p53_v19',
    'HLT_HT400_AlphaT0p51_v19',
    'HLT_HT400_AlphaT0p52_v14',
    'HLT_HT450_AlphaT0p51_v14',
    'HLT_PFNoPUHT350_PFMET100_v3',
    'HLT_PFNoPUHT400_PFMET100_v3',
    'HLT_RsqMR40_Rsq0p04_v6',
    'HLT_RsqMR45_Rsq0p09_v5',
    'HLT_RsqMR55_Rsq0p09_MR150_v6',
    'HLT_RsqMR60_Rsq0p09_MR150_v6',
    'HLT_RsqMR65_Rsq0p09_MR150_v5' ),
  HcalHPDNoise = cms.vstring( 'HLT_GlobalRunHPDNoise_v8',
    'HLT_L1Tech_HBHEHO_totalOR_v6',
    'HLT_L1Tech_HCAL_HF_single_channel_v4' ),
  HcalNZS = cms.vstring( 'HLT_HcalNZS_v10',
    'HLT_HcalPhiSym_v11' ),
  JetHT = cms.vstring( 'HLT_DiPFJetAve320_v9',
    'HLT_DiPFJetAve400_v9',
    'HLT_FatDiPFJetMass750_DR1p1_Deta1p5_v9',
    'HLT_HT200_v6',
    'HLT_HT250_v7',
    'HLT_HT300_DoubleDisplacedPFJet60_ChgFraction10_v9',
    'HLT_HT300_DoubleDisplacedPFJet60_v9',
    'HLT_HT300_SingleDisplacedPFJet60_ChgFraction10_v9',
    'HLT_HT300_SingleDisplacedPFJet60_v9',
    'HLT_HT300_v7',
    'HLT_HT350_v7',
    'HLT_HT400_v7',
    'HLT_HT450_v7',
    'HLT_HT500_v7',
    'HLT_HT550_v7',
    'HLT_HT650_Track50_dEdx3p6_v10',
    'HLT_HT650_Track60_dEdx3p7_v10',
    'HLT_HT650_v7',
    'HLT_HT750_v7',
    'HLT_Jet370_NoJetID_v15',
    'HLT_MET80_Track50_dEdx3p6_v6',
    'HLT_MET80_Track60_dEdx3p7_v6',
    'HLT_MET80_v5',
    'HLT_PFJet320_v8',
    'HLT_PFJet400_v8',
    'HLT_PFNoPUHT350_v3',
    'HLT_PFNoPUHT650_DiCentralPFNoPUJet80_CenPFNoPUJet40_v3',
    'HLT_PFNoPUHT650_v3',
    'HLT_PFNoPUHT700_v3',
    'HLT_PFNoPUHT750_v3' ),
  JetMon = cms.vstring( 'HLT_DiPFJetAve140_v9',
    'HLT_DiPFJetAve200_v9',
    'HLT_DiPFJetAve260_v9',
    'HLT_DiPFJetAve40_v8',
    'HLT_DiPFJetAve80_v9',
    'HLT_PFJet140_v8',
    'HLT_PFJet200_v8',
    'HLT_PFJet260_v8',
    'HLT_PFJet40_v7',
    'HLT_PFJet80_v8',
    'HLT_SingleForJet15_v4',
    'HLT_SingleForJet25_v4' ),
  L1Accept = cms.vstring( 'DST_Physics_v5' ),
  LogMonitor = cms.vstring( 'HLT_LogMonitor_v4' ),
  MET = cms.vstring( 'HLT_DiCentralJetSumpT100_dPhi05_DiCentralPFJet60_25_PFMET100_HBHENoiseCleaned_v4',
    'HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v4',
    'HLT_DiCentralPFJet30_PFMET80_v5',
    'HLT_DiCentralPFNoPUJet50_PFMETORPFMETNoMu80_v3',
    'HLT_DiPFJet40_PFMETnoMu65_MJJ600VBF_LeadingJets_v8',
    'HLT_DiPFJet40_PFMETnoMu65_MJJ800VBF_AllJets_v8',
    'HLT_L1ETM100_v2',
    'HLT_L1ETM30_v2',
    'HLT_L1ETM40_v2',
    'HLT_L1ETM70_v2',
    'HLT_MET120_HBHENoiseCleaned_v5',
    'HLT_MET120_v12',
    'HLT_MET200_HBHENoiseCleaned_v5',
    'HLT_MET200_v12',
    'HLT_MET300_HBHENoiseCleaned_v5',
    'HLT_MET300_v4',
    'HLT_MET400_HBHENoiseCleaned_v5',
    'HLT_MET400_v7',
    'HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v3',
    'HLT_PFMET150_v6',
    'HLT_PFMET180_v6' ),
  MinimumBias = cms.vstring( 'HLT_Physics_v5',
    'HLT_PixelTracks_Multiplicity70_v3',
    'HLT_PixelTracks_Multiplicity80_v12',
    'HLT_PixelTracks_Multiplicity90_v3',
    'HLT_Random_v2',
    'HLT_ZeroBiasPixel_DoubleTrack_v2',
    'HLT_ZeroBias_v7' ),
  MuEG = cms.vstring( 'HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v16',
    'HLT_DoubleMu8_Ele8_CaloIdT_TrkIdVL_v5',
    'HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9',
    'HLT_Mu22_Photon22_CaloIdL_v7',
    'HLT_Mu30_Ele30_CaloIdL_v8',
    'HLT_Mu7_Ele7_CaloIdT_CaloIsoVL_v7',
    'HLT_Mu8_DoubleEle8_CaloIdT_TrkIdVL_v7',
    'HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9',
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v7' ),
  MuHad = cms.vstring( 'HLT_DoubleDisplacedMu4_DiPFJet40Neutral_v7',
    'HLT_DoubleMu14_Mass8_PFMET40_v7',
    'HLT_DoubleMu14_Mass8_PFMET50_v7',
    'HLT_DoubleMu8_Mass8_PFNoPUHT175_v3',
    'HLT_DoubleMu8_Mass8_PFNoPUHT225_v3',
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT175_v3',
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT225_v3',
    'HLT_IsoMu12_DoubleCentralJet65_v4',
    'HLT_IsoMu12_RsqMR30_Rsq0p04_MR200_v4',
    'HLT_IsoMu12_RsqMR40_Rsq0p04_MR200_v4',
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_PFNoPUHT350_PFMHT40_v2',
    'HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_v7',
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7',
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7',
    'HLT_Mu40_PFNoPUHT350_v3',
    'HLT_Mu60_PFNoPUHT350_v3',
    'HLT_Mu8_DiJet30_v7',
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3',
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3',
    'HLT_Mu8_QuadJet30_v7',
    'HLT_Mu8_TriJet30_v7',
    'HLT_PFNoPUHT350_Mu15_PFMET45_v3',
    'HLT_PFNoPUHT350_Mu15_PFMET50_v3',
    'HLT_PFNoPUHT400_Mu5_PFMET45_v3',
    'HLT_PFNoPUHT400_Mu5_PFMET50_v3',
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3',
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3' ),
  MuOnia = cms.vstring( 'HLT_Dimuon0_Jpsi_Muon_v18',
    'HLT_Dimuon0_Jpsi_NoVertexing_v14',
    'HLT_Dimuon0_Jpsi_v17',
    'HLT_Dimuon0_PsiPrime_v6',
    'HLT_Dimuon0_Upsilon_Muon_v18',
    'HLT_Dimuon0_Upsilon_v17',
    'HLT_Dimuon11_Upsilon_v6',
    'HLT_Dimuon3p5_SameSign_v6',
    'HLT_Dimuon7_Upsilon_v7',
    'HLT_DoubleMu3_4_Dimuon5_Bs_Central_v5',
    'HLT_DoubleMu3p5_4_Dimuon5_Bs_Central_v5',
    'HLT_DoubleMu4_Dimuon7_Bs_Forward_v5',
    'HLT_DoubleMu4_JpsiTk_Displaced_v6',
    'HLT_DoubleMu4_Jpsi_Displaced_v12',
    'HLT_Mu5_L2Mu3_Jpsi_v7',
    'HLT_Mu5_Track2_Jpsi_v21',
    'HLT_Mu5_Track3p5_Jpsi_v7',
    'HLT_Mu7_Track7_Jpsi_v20',
    'HLT_Tau2Mu_ItTrack_v7' ),
  MuOniaParked = cms.vstring( 'HLT_BTagMu_Jet20_Mu4_v2',
    'HLT_BTagMu_Jet60_Mu4_v2',
    'HLT_Dimuon10_Jpsi_v6',
    'HLT_Dimuon5_PsiPrime_v6',
    'HLT_Dimuon5_Upsilon_v6',
    'HLT_Dimuon7_PsiPrime_v3',
    'HLT_Dimuon8_Jpsi_v7',
    'HLT_Dimuon8_Upsilon_v6',
    'HLT_DoubleMu3p5_LowMassNonResonant_Displaced_v6',
    'HLT_DoubleMu3p5_LowMass_Displaced_v6' ),
  MultiJet = cms.vstring( 'HLT_DiJet80_DiJet60_DiJet20_v5',
    'HLT_DoubleJet20_ForwardBackward_v4',
    'HLT_EightJet30_eta3p0_v5',
    'HLT_EightJet35_eta3p0_v5',
    'HLT_ExclDiJet35_HFAND_v4',
    'HLT_ExclDiJet35_HFOR_v4',
    'HLT_ExclDiJet80_HFAND_v4',
    'HLT_QuadJet60_DiJet20_v5',
    'HLT_QuadJet70_v6',
    'HLT_QuadJet80_v6',
    'HLT_QuadJet90_v6',
    'HLT_SixJet35_v6',
    'HLT_SixJet45_v6',
    'HLT_SixJet50_v6' ),
  MultiJet1Parked = cms.vstring( 'HLT_DiJet80_DiJet60_DiJet20_v5',
    'HLT_DoubleJet20_ForwardBackward_v4',
    'HLT_EightJet30_eta3p0_v5',
    'HLT_EightJet35_eta3p0_v5',
    'HLT_ExclDiJet35_HFAND_v4',
    'HLT_ExclDiJet35_HFOR_v4',
    'HLT_ExclDiJet80_HFAND_v4',
    'HLT_QuadJet50_Jet20_v4',
    'HLT_QuadJet50_v5',
    'HLT_QuadJet60_DiJet20_v5',
    'HLT_QuadJet70_v6',
    'HLT_QuadJet80_v6',
    'HLT_QuadJet90_v6',
    'HLT_SixJet35_v6',
    'HLT_SixJet45_v6',
    'HLT_SixJet50_v6' ),
  NoBPTX = cms.vstring( 'HLT_JetE30_NoBPTX3BX_NoHalo_v16',
    'HLT_JetE30_NoBPTX_v14',
    'HLT_JetE50_NoBPTX3BX_NoHalo_v13',
    'HLT_JetE70_NoBPTX3BX_NoHalo_v5',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v4',
    'HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1',
    'HLT_L2Mu20_eta2p1_NoVertex_v2',
    'HLT_L2Mu30_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1' ),
  OfflineMonitor = ( cms.vstring( 'AlCa_EcalEtaEBonly_v6',
    'AlCa_EcalEtaEEonly_v6',
    'AlCa_EcalPhiSym_v13',
    'AlCa_EcalPi0EBonly_v6',
    'AlCa_EcalPi0EEonly_v6',
    'AlCa_LumiPixels_Random_v1',
    'AlCa_LumiPixels_ZeroBias_v4',
    'AlCa_LumiPixels_v8',
    'AlCa_RPCMuonNoHits_v9',
    'AlCa_RPCMuonNoTriggers_v9',
    'AlCa_RPCMuonNormalisation_v9',
    'DST_Ele8_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT250_v4',
    'DST_HT250_v4',
    'DST_L1HTT_Or_L1MultiJet_v4',
    'DST_Mu5_HT250_v4',
    'HLT_Activity_Ecal_SC7_v13',
    'HLT_BTagMu_DiJet110_Mu5_v6',
    'HLT_BTagMu_DiJet20_Mu5_v6',
    'HLT_BTagMu_DiJet40_Mu5_v6',
    'HLT_BTagMu_DiJet70_Mu5_v6',
    'HLT_BTagMu_Jet20_Mu4_v2',
    'HLT_BTagMu_Jet300_Mu5_v6',
    'HLT_BTagMu_Jet60_Mu4_v2',
    'HLT_BeamGas_HF_Beam1_v5',
    'HLT_BeamGas_HF_Beam2_v5',
    'HLT_BeamHalo_v13',
    'HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v2',
    'HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v2',
    'HLT_CleanPFNoPUHT300_Ele40_CaloIdVT_TrkIdT_v2',
    'HLT_CleanPFNoPUHT300_Ele60_CaloIdVT_TrkIdT_v2',
    'HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v2',
    'HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v2',
    'HLT_DTErrors_v3',
    'HLT_DiCentralJetSumpT100_dPhi05_DiCentralPFJet60_25_PFMET100_HBHENoiseCleaned_v4',
    'HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v4',
    'HLT_DiCentralPFJet30_PFMET80_v5',
    'HLT_DiCentralPFNoPUJet50_PFMETORPFMETNoMu80_v3',
    'HLT_DiJet35_MJJ650_AllJets_DEta3p5_VBF_v5',
    'HLT_DiJet35_MJJ700_AllJets_DEta3p5_VBF_v5',
    'HLT_DiJet35_MJJ750_AllJets_DEta3p5_VBF_v5',
    'HLT_DiJet40Eta2p6_BTagIP3DFastPV_v7',
    'HLT_DiJet80Eta2p6_BTagIP3DFastPVLoose_v7',
    'HLT_DiJet80_DiJet60_DiJet20_v5',
    'HLT_DiPFJet40_PFMETnoMu65_MJJ600VBF_LeadingJets_v8',
    'HLT_DiPFJet40_PFMETnoMu65_MJJ800VBF_AllJets_v8',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05_v4',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d03_v4',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d05_v4',
    'HLT_DiPFJetAve140_v9',
    'HLT_DiPFJetAve200_v9',
    'HLT_DiPFJetAve260_v9',
    'HLT_DiPFJetAve320_v9',
    'HLT_DiPFJetAve400_v9',
    'HLT_DiPFJetAve40_v8',
    'HLT_DiPFJetAve80_v9',
    'HLT_Dimuon0_Jpsi_Muon_v18',
    'HLT_Dimuon0_Jpsi_NoVertexing_v14',
    'HLT_Dimuon0_Jpsi_v17',
    'HLT_Dimuon0_PsiPrime_v6',
    'HLT_Dimuon0_Upsilon_Muon_v18',
    'HLT_Dimuon0_Upsilon_v17',
    'HLT_Dimuon10_Jpsi_v6',
    'HLT_Dimuon11_Upsilon_v6',
    'HLT_Dimuon3p5_SameSign_v6',
    'HLT_Dimuon5_PsiPrime_v6',
    'HLT_Dimuon5_Upsilon_v6',
    'HLT_Dimuon7_PsiPrime_v3',
    'HLT_Dimuon7_Upsilon_v7',
    'HLT_Dimuon8_Jpsi_v7',
    'HLT_Dimuon8_Upsilon_v6',
    'HLT_DisplacedPhoton65EBOnly_CaloIdVL_IsoL_PFMET30_v3',
    'HLT_DisplacedPhoton65_CaloIdVL_IsoL_PFMET25_v3',
    'HLT_DoubleDisplacedMu4_DiPFJet40Neutral_v7',
    'HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v12',
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7',
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7',
    'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v7',
    'HLT_DoubleEle33_CaloIdL_v14',
    'HLT_DoubleEle33_CaloIdT_v10',
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3',
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3',
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_v12',
    'HLT_DoubleJet20_ForwardBackward_v4',
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v4',
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v3',
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v3',
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v3',
    'HLT_DoubleMu11_Acoplanarity03_v5',
    'HLT_DoubleMu14_Mass8_PFMET40_v7',
    'HLT_DoubleMu14_Mass8_PFMET50_v7',
    'HLT_DoubleMu3_4_Dimuon5_Bs_Central_v5',
    'HLT_DoubleMu3p5_4_Dimuon5_Bs_Central_v5',
    'HLT_DoubleMu3p5_LowMassNonResonant_Displaced_v6',
    'HLT_DoubleMu3p5_LowMass_Displaced_v6',
    'HLT_DoubleMu4_Acoplanarity03_v5',
    'HLT_DoubleMu4_Dimuon7_Bs_Forward_v5',
    'HLT_DoubleMu4_JpsiTk_Displaced_v6',
    'HLT_DoubleMu4_Jpsi_Displaced_v12',
    'HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v16',
    'HLT_DoubleMu5_IsoMu5_v20',
    'HLT_DoubleMu8_Ele8_CaloIdT_TrkIdVL_v5',
    'HLT_DoubleMu8_Mass8_PFNoPUHT175_v3',
    'HLT_DoubleMu8_Mass8_PFNoPUHT225_v3',
    'HLT_DoublePhoton40_CaloIdL_Rsq0p035_v6',
    'HLT_DoublePhoton40_CaloIdL_Rsq0p06_v6',
    'HLT_DoublePhoton48_HEVT_v8',
    'HLT_DoublePhoton53_HEVT_v2',
    'HLT_DoublePhoton5_IsoVL_CEP_v16',
    'HLT_DoublePhoton70_v6',
    'HLT_DoublePhoton80_v7',
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT175_v3',
    'HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT225_v3',
    'HLT_EightJet30_eta3p0_v5',
    'HLT_EightJet35_eta3p0_v5',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_DoubleCentralJet65_v4',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR30_Rsq0p04_MR200_v4',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR40_Rsq0p04_MR200_v4',
    'HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v6',
    'HLT_Ele17_CaloIdL_CaloIsoVL_v17',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v18',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v6',
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass50_v6',
    'HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v6',
    'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_Ele22_eta2p1_WP90NoIso_LooseIsoPFTau20_v6',
    'HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v6',
    'HLT_Ele23_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT30_v7',
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_BTagIPIter_v8',
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_v7',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_DiCentralPFNoPUJet30_v1',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet30_v3',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet50_40_30_v3',
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v7',
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT15_v7',
    'HLT_Ele27_WP80_CentralPFJet80_v8',
    'HLT_Ele27_WP80_PFMET_MT50_v6',
    'HLT_Ele27_WP80_WCandPt80_v8',
    'HLT_Ele27_WP80_v11',
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet100_PFNoPUJet25_v7',
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet150_PFNoPUJet25_v7',
    'HLT_Ele30_CaloIdVT_TrkIdT_v6',
    'HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_Mass50_v6',
    'HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_PFMET20_v3',
    'HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_v3',
    'HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_CentralPFJet30_v3',
    'HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_v3',
    'HLT_Ele5_SC5_Jpsi_Mass2to15_v4',
    'HLT_Ele80_CaloIdVT_GsfTrkIdT_v2',
    'HLT_Ele8_CaloIdL_CaloIsoVL_v17',
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7',
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v15',
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdVL_EG7_v2',
    'HLT_Ele8_CaloIdT_TrkIdVL_Jet30_v7',
    'HLT_Ele8_CaloIdT_TrkIdVL_v5',
    'HLT_Ele90_CaloIdVT_GsfTrkIdT_v2',
    'HLT_ExclDiJet35_HFAND_v4',
    'HLT_ExclDiJet35_HFOR_v4',
    'HLT_ExclDiJet80_HFAND_v4',
    'HLT_FatDiPFJetMass750_DR1p1_Deta1p5_v9',
    'HLT_GlobalRunHPDNoise_v8',
    'HLT_HT200_AlphaT0p57_v8',
    'HLT_HT200_v6',
    'HLT_HT250_AlphaT0p55_v8',
    'HLT_HT250_AlphaT0p57_v8',
    'HLT_HT250_v7',
    'HLT_HT300_AlphaT0p53_v8',
    'HLT_HT300_AlphaT0p54_v14',
    'HLT_HT300_DoubleDisplacedPFJet60_ChgFraction10_v9',
    'HLT_HT300_DoubleDisplacedPFJet60_v9',
    'HLT_HT300_SingleDisplacedPFJet60_ChgFraction10_v9',
    'HLT_HT300_SingleDisplacedPFJet60_v9',
    'HLT_HT300_v7',
    'HLT_HT350_AlphaT0p52_v8',
    'HLT_HT350_AlphaT0p53_v19',
    'HLT_HT350_v7',
    'HLT_HT400_AlphaT0p51_v19',
    'HLT_HT400_AlphaT0p52_v14',
    'HLT_HT400_v7',
    'HLT_HT450_AlphaT0p51_v14',
    'HLT_HT450_v7',
    'HLT_HT500_v7',
    'HLT_HT550_v7',
    'HLT_HT650_Track50_dEdx3p6_v10',
    'HLT_HT650_Track60_dEdx3p7_v10',
    'HLT_HT650_v7',
    'HLT_HT750_v7',
    'HLT_HcalCalibration_v3',
    'HLT_HcalNZS_v10',
    'HLT_HcalPhiSym_v11',
    'HLT_IsoMu12_DoubleCentralJet65_v4',
    'HLT_IsoMu12_RsqMR30_Rsq0p04_MR200_v4',
    'HLT_IsoMu12_RsqMR40_Rsq0p04_MR200_v4',
    'HLT_IsoMu15_eta2p1_L1ETM20_v7',
    'HLT_IsoMu15_eta2p1_LooseIsoPFTau35_Trk20_Prong1_L1ETM20_v9',
    'HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3',
    'HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_PFNoPUHT350_PFMHT40_v2',
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v6',
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_IsoMu18_eta2p1_MediumIsoPFTau25_Trk1_eta2p1_v3',
    'HLT_IsoMu20_WCandPt80_v3',
    'HLT_IsoMu20_eta2p1_CentralPFJet80_v8',
    'HLT_IsoMu20_eta2p1_v7',
    'HLT_IsoMu24_CentralPFJet30_CentralPFJet25_PFMET20_v3',
    'HLT_IsoMu24_CentralPFJet30_CentralPFJet25_v3',
    'HLT_IsoMu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3',
    'HLT_IsoMu24_eta2p1_v15',
    'HLT_IsoMu24_v17',
    'HLT_IsoMu30_eta2p1_v15',
    'HLT_IsoMu30_v11',
    'HLT_IsoMu34_eta2p1_v13',
    'HLT_IsoMu40_eta2p1_v10',
    'HLT_IsoTrackHB_v14',
    'HLT_IsoTrackHE_v15',
    'HLT_Jet160Eta2p4_Jet120Eta2p4_DiBTagIP3DFastPVLoose_v7',
    'HLT_Jet370_NoJetID_v15',
    'HLT_Jet60Eta1p7_Jet53Eta1p7_DiBTagIP3DFastPV_v7',
    'HLT_Jet80Eta1p7_Jet70Eta1p7_DiBTagIP3DFastPV_v7',
    'HLT_JetE30_NoBPTX3BX_NoHalo_v16',
    'HLT_JetE30_NoBPTX_v14',
    'HLT_JetE50_NoBPTX3BX_NoHalo_v13',
    'HLT_JetE70_NoBPTX3BX_NoHalo_v5',
    'HLT_L1DoubleEG3_FwdVeto_v2',
    'HLT_L1DoubleJet36Central_v7',
    'HLT_L1ETM100_v2',
    'HLT_L1ETM30_v2',
    'HLT_L1ETM40_v2',
    'HLT_L1ETM70_v2',
    'HLT_L1SingleEG12_v6',
    'HLT_L1SingleEG5_v6',
    'HLT_L1SingleJet16_v7',
    'HLT_L1SingleJet36_v7',
    'HLT_L1SingleMu12_v2',
    'HLT_L1SingleMuOpen_AntiBPTX_v7',
    'HLT_L1SingleMuOpen_v7',
    'HLT_L1Tech_HBHEHO_totalOR_v6',
    'HLT_L1Tech_HCAL_HF_single_channel_v4',
    'HLT_L1TrackerCosmics_v7',
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v3',
    'HLT_L2DoubleMu23_NoVertex_v11',
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v3',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v4',
    'HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1',
    'HLT_L2Mu20_eta2p1_NoVertex_v2',
    'HLT_L2Mu30_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1',
    'HLT_L2Mu70_2Cha_eta2p1_PFMET55_v1')+cms.vstring( 'HLT_L2Mu70_2Cha_eta2p1_PFMET60_v1',
    'HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_v7',
    'HLT_LogMonitor_v4',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_v9',
    'HLT_MET120_HBHENoiseCleaned_v5',
    'HLT_MET120_v12',
    'HLT_MET200_HBHENoiseCleaned_v5',
    'HLT_MET200_v12',
    'HLT_MET300_HBHENoiseCleaned_v5',
    'HLT_MET300_v4',
    'HLT_MET400_HBHENoiseCleaned_v5',
    'HLT_MET400_v7',
    'HLT_MET80_Track50_dEdx3p6_v6',
    'HLT_MET80_Track60_dEdx3p7_v6',
    'HLT_MET80_v5',
    'HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v3',
    'HLT_Mu12_eta2p1_DiCentral_20_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_BTagIP3D1stTrack_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_DiBTagIP3D1stTrack_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_v7',
    'HLT_Mu12_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v6',
    'HLT_Mu12_v18',
    'HLT_Mu13_Mu8_v21',
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7',
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7',
    'HLT_Mu15_eta2p1_L1ETM20_v5',
    'HLT_Mu15_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v2',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_BTagIP3D1stTrack_v7',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_DiBTagIP3D1stTrack_v7',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_v7',
    'HLT_Mu15_eta2p1_v5',
    'HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9',
    'HLT_Mu17_Mu8_v21',
    'HLT_Mu17_TkMu8_v13',
    'HLT_Mu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3',
    'HLT_Mu17_eta2p1_LooseIsoPFTau20_v6',
    'HLT_Mu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_Mu17_v5',
    'HLT_Mu22_Photon22_CaloIdL_v7',
    'HLT_Mu22_TkMu22_v8',
    'HLT_Mu22_TkMu8_v8',
    'HLT_Mu24_CentralPFJet30_CentralPFJet25_v3',
    'HLT_Mu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3',
    'HLT_Mu24_eta2p1_v5',
    'HLT_Mu24_v16',
    'HLT_Mu30_Ele30_CaloIdL_v8',
    'HLT_Mu30_eta2p1_v5',
    'HLT_Mu30_v16',
    'HLT_Mu40_PFNoPUHT350_v3',
    'HLT_Mu40_eta2p1_Track50_dEdx3p6_v5',
    'HLT_Mu40_eta2p1_Track60_dEdx3p7_v5',
    'HLT_Mu40_eta2p1_v11',
    'HLT_Mu40_v14',
    'HLT_Mu50_eta2p1_v8',
    'HLT_Mu5_L2Mu3_Jpsi_v7',
    'HLT_Mu5_Track2_Jpsi_v21',
    'HLT_Mu5_Track3p5_Jpsi_v7',
    'HLT_Mu5_v20',
    'HLT_Mu60_PFNoPUHT350_v3',
    'HLT_Mu7_Ele7_CaloIdT_CaloIsoVL_v7',
    'HLT_Mu7_Track7_Jpsi_v20',
    'HLT_Mu8_DiJet30_v7',
    'HLT_Mu8_DoubleEle8_CaloIdT_TrkIdVL_v7',
    'HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9',
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v7',
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3',
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3',
    'HLT_Mu8_QuadJet30_v7',
    'HLT_Mu8_TriJet30_v7',
    'HLT_Mu8_v18',
    'HLT_PFJet140_v8',
    'HLT_PFJet200_v8',
    'HLT_PFJet260_v8',
    'HLT_PFJet320_v8',
    'HLT_PFJet400_v8',
    'HLT_PFJet40_v7',
    'HLT_PFJet80_v8',
    'HLT_PFMET150_v6',
    'HLT_PFMET180_v6',
    'HLT_PFNoPUHT350_Mu15_PFMET45_v3',
    'HLT_PFNoPUHT350_Mu15_PFMET50_v3',
    'HLT_PFNoPUHT350_PFMET100_v3',
    'HLT_PFNoPUHT350_v3',
    'HLT_PFNoPUHT400_Mu5_PFMET45_v3',
    'HLT_PFNoPUHT400_Mu5_PFMET50_v3',
    'HLT_PFNoPUHT400_PFMET100_v3',
    'HLT_PFNoPUHT650_DiCentralPFNoPUJet80_CenPFNoPUJet40_v3',
    'HLT_PFNoPUHT650_v3',
    'HLT_PFNoPUHT700_v3',
    'HLT_PFNoPUHT750_v3',
    'HLT_Photon135_v7',
    'HLT_Photon150_v4',
    'HLT_Photon160_v4',
    'HLT_Photon20_CaloIdVL_IsoL_v16',
    'HLT_Photon20_CaloIdVL_v4',
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon26_CaloId10_Iso50_Photon18_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_CaloId10_Iso50_Photon18_R9Id85_Mass60_v6',
    'HLT_Photon26_Photon18_v12',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass70_v2',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_v5',
    'HLT_Photon26_R9Id85_Photon18_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_R9Id85_Photon18_R9Id85_Mass60_v4',
    'HLT_Photon300_NoHE_v5',
    'HLT_Photon30_CaloIdVL_v14',
    'HLT_Photon36_CaloId10_Iso50_Photon22_CaloId10_Iso50_v6',
    'HLT_Photon36_CaloId10_Iso50_Photon22_R9Id85_v6',
    'HLT_Photon36_Photon22_v6',
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v6',
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_v5',
    'HLT_Photon36_R9Id85_Photon22_CaloId10_Iso50_v6',
    'HLT_Photon36_R9Id85_Photon22_R9Id85_v4',
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon40_CaloIdL_RsqMR40_Rsq0p09_MR150_v6',
    'HLT_Photon40_CaloIdL_RsqMR45_Rsq0p09_MR150_v6',
    'HLT_Photon40_CaloIdL_RsqMR50_Rsq0p09_MR150_v6',
    'HLT_Photon50_CaloIdVL_IsoL_v17',
    'HLT_Photon50_CaloIdVL_v10',
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon60_CaloIdL_HT300_v4',
    'HLT_Photon60_CaloIdL_MHT70_v11',
    'HLT_Photon70_CaloIdXL_PFMET100_v6',
    'HLT_Photon70_CaloIdXL_PFNoPUHT400_v3',
    'HLT_Photon70_CaloIdXL_PFNoPUHT500_v3',
    'HLT_Photon75_CaloIdVL_v13',
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon90_CaloIdVL_v10',
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Physics_v5',
    'HLT_PixelTracks_Multiplicity70_v3',
    'HLT_PixelTracks_Multiplicity80_v12',
    'HLT_PixelTracks_Multiplicity90_v3',
    'HLT_QuadJet50_Jet20_v4',
    'HLT_QuadJet50_v5',
    'HLT_QuadJet60_DiJet20_v5',
    'HLT_QuadJet70_v6',
    'HLT_QuadJet75_55_35_20_BTagIP_VBF_v6',
    'HLT_QuadJet75_55_38_20_BTagIP_VBF_v6',
    'HLT_QuadJet80_v6',
    'HLT_QuadJet90_v6',
    'HLT_QuadPFJet78_61_44_31_BTagCSV_VBF_v4',
    'HLT_QuadPFJet82_65_48_35_BTagCSV_VBF_v4',
    'HLT_Random_v2',
    'HLT_RelIso1p0Mu20_v3',
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3',
    'HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3',
    'HLT_RelIso1p0Mu5_v6',
    'HLT_RsqMR40_Rsq0p04_v6',
    'HLT_RsqMR45_Rsq0p09_v5',
    'HLT_RsqMR55_Rsq0p09_MR150_v6',
    'HLT_RsqMR60_Rsq0p09_MR150_v6',
    'HLT_RsqMR65_Rsq0p09_MR150_v5',
    'HLT_SingleForJet15_v4',
    'HLT_SingleForJet25_v4',
    'HLT_SixJet35_v6',
    'HLT_SixJet45_v6',
    'HLT_SixJet50_v6',
    'HLT_Tau2Mu_ItTrack_v7',
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v18',
    'HLT_TripleMu5_v19',
    'HLT_ZeroBiasPixel_DoubleTrack_v2',
    'HLT_ZeroBias_v7') ),
  OnlineHltMonitor = cms.vstring( 'HLT_DiJet80_DiJet60_DiJet20_v5',
    'HLT_DiPFJetAve140_v9',
    'HLT_DiPFJetAve200_v9',
    'HLT_DiPFJetAve260_v9',
    'HLT_DiPFJetAve320_v9',
    'HLT_DiPFJetAve400_v9',
    'HLT_DiPFJetAve40_v8',
    'HLT_DiPFJetAve80_v9',
    'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele27_WP80_PFMET_MT50_v6',
    'HLT_Ele27_WP80_v11',
    'HLT_Ele30_CaloIdVT_TrkIdT_v6',
    'HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele80_CaloIdVT_GsfTrkIdT_v2',
    'HLT_Ele90_CaloIdVT_GsfTrkIdT_v2',
    'HLT_IsoMu20_eta2p1_v7',
    'HLT_IsoMu24_eta2p1_v15',
    'HLT_IsoMu30_eta2p1_v15',
    'HLT_IsoMu34_eta2p1_v13',
    'HLT_IsoMu40_eta2p1_v10',
    'HLT_Jet370_NoJetID_v15',
    'HLT_Mu12_v18',
    'HLT_Mu15_eta2p1_v5',
    'HLT_Mu17_v5',
    'HLT_Mu24_eta2p1_v5',
    'HLT_Mu30_eta2p1_v5',
    'HLT_Mu40_eta2p1_Track50_dEdx3p6_v5',
    'HLT_Mu40_eta2p1_Track60_dEdx3p7_v5',
    'HLT_Mu40_eta2p1_v11',
    'HLT_Mu5_v20',
    'HLT_Mu8_v18',
    'HLT_PFJet140_v8',
    'HLT_PFJet200_v8',
    'HLT_PFJet260_v8',
    'HLT_PFJet320_v8',
    'HLT_PFJet400_v8',
    'HLT_PFJet40_v7',
    'HLT_PFJet80_v8',
    'HLT_RelIso1p0Mu20_v3',
    'HLT_RelIso1p0Mu5_v6',
    'HLT_SingleForJet15_v4',
    'HLT_SingleForJet25_v4' ),
  OnlineMonitor = ( cms.vstring( 'HLT_Activity_Ecal_SC7_v13',
    'HLT_BTagMu_DiJet110_Mu5_v6',
    'HLT_BTagMu_DiJet20_Mu5_v6',
    'HLT_BTagMu_DiJet40_Mu5_v6',
    'HLT_BTagMu_DiJet70_Mu5_v6',
    'HLT_BTagMu_Jet300_Mu5_v6',
    'HLT_BeamGas_HF_Beam1_v5',
    'HLT_BeamGas_HF_Beam2_v5',
    'HLT_BeamHalo_v13',
    'HLT_DTErrors_v3',
    'HLT_DiCentralJetSumpT100_dPhi05_DiCentralPFJet60_25_PFMET100_HBHENoiseCleaned_v4',
    'HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v4',
    'HLT_DiCentralPFJet30_PFMET80_v5',
    'HLT_DiJet40Eta2p6_BTagIP3DFastPV_v7',
    'HLT_DiJet80Eta2p6_BTagIP3DFastPVLoose_v7',
    'HLT_DiJet80_DiJet60_DiJet20_v5',
    'HLT_DiPFJet40_PFMETnoMu65_MJJ600VBF_LeadingJets_v8',
    'HLT_DiPFJet40_PFMETnoMu65_MJJ800VBF_AllJets_v8',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05_v4',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d03_v4',
    'HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d05_v4',
    'HLT_DiPFJetAve140_v9',
    'HLT_DiPFJetAve200_v9',
    'HLT_DiPFJetAve260_v9',
    'HLT_DiPFJetAve320_v9',
    'HLT_DiPFJetAve400_v9',
    'HLT_DiPFJetAve40_v8',
    'HLT_DiPFJetAve80_v9',
    'HLT_Dimuon0_Jpsi_Muon_v18',
    'HLT_Dimuon0_Jpsi_NoVertexing_v14',
    'HLT_Dimuon0_Jpsi_v17',
    'HLT_Dimuon0_PsiPrime_v6',
    'HLT_Dimuon0_Upsilon_Muon_v18',
    'HLT_Dimuon0_Upsilon_v17',
    'HLT_Dimuon11_Upsilon_v6',
    'HLT_Dimuon3p5_SameSign_v6',
    'HLT_Dimuon7_Upsilon_v7',
    'HLT_DisplacedPhoton65EBOnly_CaloIdVL_IsoL_PFMET30_v3',
    'HLT_DisplacedPhoton65_CaloIdVL_IsoL_PFMET25_v3',
    'HLT_DoubleDisplacedMu4_DiPFJet40Neutral_v7',
    'HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v12',
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7',
    'HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7',
    'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v7',
    'HLT_DoubleEle33_CaloIdL_v14',
    'HLT_DoubleEle33_CaloIdT_v10',
    'HLT_DoubleEle8_CaloIdT_TrkIdVL_v12',
    'HLT_DoubleJet20_ForwardBackward_v4',
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v4',
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v3',
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v3',
    'HLT_DoubleMu11_Acoplanarity03_v5',
    'HLT_DoubleMu14_Mass8_PFMET40_v7',
    'HLT_DoubleMu14_Mass8_PFMET50_v7',
    'HLT_DoubleMu3_4_Dimuon5_Bs_Central_v5',
    'HLT_DoubleMu3p5_4_Dimuon5_Bs_Central_v5',
    'HLT_DoubleMu4_Acoplanarity03_v5',
    'HLT_DoubleMu4_Dimuon7_Bs_Forward_v5',
    'HLT_DoubleMu4_JpsiTk_Displaced_v6',
    'HLT_DoubleMu4_Jpsi_Displaced_v12',
    'HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v16',
    'HLT_DoubleMu5_IsoMu5_v20',
    'HLT_DoubleMu8_Ele8_CaloIdT_TrkIdVL_v5',
    'HLT_DoublePhoton40_CaloIdL_Rsq0p035_v6',
    'HLT_DoublePhoton40_CaloIdL_Rsq0p06_v6',
    'HLT_DoublePhoton48_HEVT_v8',
    'HLT_DoublePhoton53_HEVT_v2',
    'HLT_DoublePhoton5_IsoVL_CEP_v16',
    'HLT_DoublePhoton70_v6',
    'HLT_DoublePhoton80_v7',
    'HLT_EightJet30_eta3p0_v5',
    'HLT_EightJet35_eta3p0_v5',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_DoubleCentralJet65_v4',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR30_Rsq0p04_MR200_v4',
    'HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR40_Rsq0p04_MR200_v4',
    'HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v6',
    'HLT_Ele17_CaloIdL_CaloIsoVL_v17',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v18',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7',
    'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v6',
    'HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass50_v6',
    'HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v6',
    'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_Ele22_eta2p1_WP90NoIso_LooseIsoPFTau20_v6',
    'HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v6',
    'HLT_Ele23_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT30_v7',
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_BTagIPIter_v8',
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_v7',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_DiCentralPFNoPUJet30_v1',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet30_v3',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet50_40_30_v3',
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v7',
    'HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT15_v7',
    'HLT_Ele27_WP80_CentralPFJet80_v8',
    'HLT_Ele27_WP80_PFMET_MT50_v6',
    'HLT_Ele27_WP80_WCandPt80_v8',
    'HLT_Ele27_WP80_v11',
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet100_PFNoPUJet25_v7',
    'HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet150_PFNoPUJet25_v7',
    'HLT_Ele30_CaloIdVT_TrkIdT_v6',
    'HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_Mass50_v6',
    'HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_PFMET20_v3',
    'HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_v3',
    'HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_CentralPFJet30_v3',
    'HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_v3',
    'HLT_Ele5_SC5_Jpsi_Mass2to15_v4',
    'HLT_Ele80_CaloIdVT_GsfTrkIdT_v2',
    'HLT_Ele8_CaloIdL_CaloIsoVL_v17',
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7',
    'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v15',
    'HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v18',
    'HLT_Ele8_CaloIdT_TrkIdVL_EG7_v2',
    'HLT_Ele8_CaloIdT_TrkIdVL_Jet30_v7',
    'HLT_Ele8_CaloIdT_TrkIdVL_v5',
    'HLT_Ele90_CaloIdVT_GsfTrkIdT_v2',
    'HLT_ExclDiJet35_HFAND_v4',
    'HLT_ExclDiJet35_HFOR_v4',
    'HLT_ExclDiJet80_HFAND_v4',
    'HLT_FatDiPFJetMass750_DR1p1_Deta1p5_v9',
    'HLT_GlobalRunHPDNoise_v8',
    'HLT_HT200_v6',
    'HLT_HT250_AlphaT0p55_v8',
    'HLT_HT250_AlphaT0p57_v8',
    'HLT_HT250_v7',
    'HLT_HT300_AlphaT0p53_v8',
    'HLT_HT300_AlphaT0p54_v14',
    'HLT_HT300_DoubleDisplacedPFJet60_ChgFraction10_v9',
    'HLT_HT300_DoubleDisplacedPFJet60_v9',
    'HLT_HT300_SingleDisplacedPFJet60_ChgFraction10_v9',
    'HLT_HT300_SingleDisplacedPFJet60_v9',
    'HLT_HT300_v7',
    'HLT_HT350_AlphaT0p52_v8',
    'HLT_HT350_AlphaT0p53_v19',
    'HLT_HT350_v7',
    'HLT_HT400_AlphaT0p51_v19',
    'HLT_HT400_AlphaT0p52_v14',
    'HLT_HT400_v7',
    'HLT_HT450_AlphaT0p51_v14',
    'HLT_HT450_v7',
    'HLT_HT500_v7',
    'HLT_HT550_v7',
    'HLT_HT650_Track50_dEdx3p6_v10',
    'HLT_HT650_Track60_dEdx3p7_v10',
    'HLT_HT650_v7',
    'HLT_HT750_v7',
    'HLT_HcalNZS_v10',
    'HLT_HcalPhiSym_v11',
    'HLT_IsoMu12_DoubleCentralJet65_v4',
    'HLT_IsoMu12_RsqMR30_Rsq0p04_MR200_v4',
    'HLT_IsoMu12_RsqMR40_Rsq0p04_MR200_v4',
    'HLT_IsoMu15_eta2p1_L1ETM20_v7',
    'HLT_IsoMu15_eta2p1_LooseIsoPFTau35_Trk20_Prong1_L1ETM20_v9',
    'HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3',
    'HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v6',
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_IsoMu20_WCandPt80_v3',
    'HLT_IsoMu20_eta2p1_CentralPFJet80_v8',
    'HLT_IsoMu20_eta2p1_v7',
    'HLT_IsoMu24_CentralPFJet30_CentralPFJet25_PFMET20_v3',
    'HLT_IsoMu24_CentralPFJet30_CentralPFJet25_v3',
    'HLT_IsoMu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3',
    'HLT_IsoMu24_eta2p1_v15',
    'HLT_IsoMu24_v17',
    'HLT_IsoMu30_eta2p1_v15',
    'HLT_IsoMu30_v11',
    'HLT_IsoMu34_eta2p1_v13',
    'HLT_IsoMu40_eta2p1_v10',
    'HLT_IsoTrackHB_v14',
    'HLT_IsoTrackHE_v15',
    'HLT_Jet160Eta2p4_Jet120Eta2p4_DiBTagIP3DFastPVLoose_v7',
    'HLT_Jet370_NoJetID_v15',
    'HLT_Jet60Eta1p7_Jet53Eta1p7_DiBTagIP3DFastPV_v7',
    'HLT_Jet80Eta1p7_Jet70Eta1p7_DiBTagIP3DFastPV_v7',
    'HLT_JetE30_NoBPTX3BX_NoHalo_v16',
    'HLT_JetE30_NoBPTX_v14',
    'HLT_JetE50_NoBPTX3BX_NoHalo_v13',
    'HLT_JetE70_NoBPTX3BX_NoHalo_v5',
    'HLT_L1DoubleEG3_FwdVeto_v2',
    'HLT_L1DoubleJet36Central_v7',
    'HLT_L1ETM100_v2',
    'HLT_L1ETM30_v2',
    'HLT_L1ETM40_v2',
    'HLT_L1ETM70_v2',
    'HLT_L1SingleEG12_v6',
    'HLT_L1SingleEG5_v6',
    'HLT_L1SingleJet16_v7',
    'HLT_L1SingleJet36_v7',
    'HLT_L1SingleMu12_v2',
    'HLT_L1SingleMuOpen_AntiBPTX_v7',
    'HLT_L1SingleMuOpen_v7',
    'HLT_L1Tech_HBHEHO_totalOR_v6',
    'HLT_L1Tech_HCAL_HF_single_channel_v4',
    'HLT_L1TrackerCosmics_v7',
    'HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v3',
    'HLT_L2DoubleMu23_NoVertex_v11',
    'HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v3',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v4',
    'HLT_L2Mu20_eta2p1_NoVertex_v2',
    'HLT_L2Mu70_2Cha_eta2p1_PFMET55_v1',
    'HLT_L2Mu70_2Cha_eta2p1_PFMET60_v1',
    'HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_v7',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_v9',
    'HLT_MET120_HBHENoiseCleaned_v5',
    'HLT_MET120_v12',
    'HLT_MET200_HBHENoiseCleaned_v5',
    'HLT_MET200_v12',
    'HLT_MET300_HBHENoiseCleaned_v5',
    'HLT_MET300_v4',
    'HLT_MET400_HBHENoiseCleaned_v5',
    'HLT_MET400_v7',
    'HLT_MET80_Track50_dEdx3p6_v6',
    'HLT_MET80_Track60_dEdx3p7_v6',
    'HLT_MET80_v5',
    'HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v3',
    'HLT_Mu12_eta2p1_DiCentral_20_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_BTagIP3D1stTrack_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_DiBTagIP3D1stTrack_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_v7',
    'HLT_Mu12_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v6',
    'HLT_Mu12_v18',
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7',
    'HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7',
    'HLT_Mu15_eta2p1_L1ETM20_v5',
    'HLT_Mu15_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v2',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_BTagIP3D1stTrack_v7',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_DiBTagIP3D1stTrack_v7',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_v7',
    'HLT_Mu15_eta2p1_v5',
    'HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9',
    'HLT_Mu17_Mu8_v21',
    'HLT_Mu17_TkMu8_v13',
    'HLT_Mu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3',
    'HLT_Mu17_eta2p1_LooseIsoPFTau20_v6',
    'HLT_Mu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_Mu17_v5',
    'HLT_Mu22_Photon22_CaloIdL_v7',
    'HLT_Mu22_TkMu22_v8',
    'HLT_Mu22_TkMu8_v8',
    'HLT_Mu24_CentralPFJet30_CentralPFJet25_v3',
    'HLT_Mu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3',
    'HLT_Mu24_eta2p1_v5',
    'HLT_Mu24_v16',
    'HLT_Mu30_Ele30_CaloIdL_v8',
    'HLT_Mu30_eta2p1_v5',
    'HLT_Mu30_v16')+cms.vstring( 'HLT_Mu40_eta2p1_Track50_dEdx3p6_v5',
    'HLT_Mu40_eta2p1_Track60_dEdx3p7_v5',
    'HLT_Mu40_eta2p1_v11',
    'HLT_Mu40_v14',
    'HLT_Mu50_eta2p1_v8',
    'HLT_Mu5_L2Mu3_Jpsi_v7',
    'HLT_Mu5_Track2_Jpsi_v21',
    'HLT_Mu5_Track3p5_Jpsi_v7',
    'HLT_Mu5_v20',
    'HLT_Mu7_Ele7_CaloIdT_CaloIsoVL_v7',
    'HLT_Mu7_Track7_Jpsi_v20',
    'HLT_Mu8_DiJet30_v7',
    'HLT_Mu8_DoubleEle8_CaloIdT_TrkIdVL_v7',
    'HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9',
    'HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v7',
    'HLT_Mu8_QuadJet30_v7',
    'HLT_Mu8_TriJet30_v7',
    'HLT_Mu8_v18',
    'HLT_PFJet140_v8',
    'HLT_PFJet200_v8',
    'HLT_PFJet260_v8',
    'HLT_PFJet320_v8',
    'HLT_PFJet400_v8',
    'HLT_PFJet40_v7',
    'HLT_PFJet80_v8',
    'HLT_PFMET150_v6',
    'HLT_PFMET180_v6',
    'HLT_Photon135_v7',
    'HLT_Photon150_v4',
    'HLT_Photon160_v4',
    'HLT_Photon20_CaloIdVL_IsoL_v16',
    'HLT_Photon20_CaloIdVL_v4',
    'HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon26_CaloId10_Iso50_Photon18_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_CaloId10_Iso50_Photon18_R9Id85_Mass60_v6',
    'HLT_Photon26_Photon18_v12',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass70_v2',
    'HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_v5',
    'HLT_Photon26_R9Id85_Photon18_CaloId10_Iso50_Mass60_v6',
    'HLT_Photon26_R9Id85_Photon18_R9Id85_Mass60_v4',
    'HLT_Photon300_NoHE_v5',
    'HLT_Photon30_CaloIdVL_v14',
    'HLT_Photon36_CaloId10_Iso50_Photon22_CaloId10_Iso50_v6',
    'HLT_Photon36_CaloId10_Iso50_Photon22_R9Id85_v6',
    'HLT_Photon36_Photon22_v6',
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v6',
    'HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_v5',
    'HLT_Photon36_R9Id85_Photon22_CaloId10_Iso50_v6',
    'HLT_Photon36_R9Id85_Photon22_R9Id85_v4',
    'HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon40_CaloIdL_RsqMR40_Rsq0p09_MR150_v6',
    'HLT_Photon40_CaloIdL_RsqMR45_Rsq0p09_MR150_v6',
    'HLT_Photon40_CaloIdL_RsqMR50_Rsq0p09_MR150_v6',
    'HLT_Photon50_CaloIdVL_IsoL_v17',
    'HLT_Photon50_CaloIdVL_v10',
    'HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon60_CaloIdL_HT300_v4',
    'HLT_Photon60_CaloIdL_MHT70_v11',
    'HLT_Photon70_CaloIdXL_PFMET100_v6',
    'HLT_Photon75_CaloIdVL_v13',
    'HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Photon90_CaloIdVL_v10',
    'HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_v5',
    'HLT_Physics_v5',
    'HLT_PixelTracks_Multiplicity70_v3',
    'HLT_PixelTracks_Multiplicity80_v12',
    'HLT_PixelTracks_Multiplicity90_v3',
    'HLT_QuadJet60_DiJet20_v5',
    'HLT_QuadJet70_v6',
    'HLT_QuadJet75_55_35_20_BTagIP_VBF_v6',
    'HLT_QuadJet75_55_38_20_BTagIP_VBF_v6',
    'HLT_QuadJet80_v6',
    'HLT_QuadJet90_v6',
    'HLT_QuadPFJet78_61_44_31_BTagCSV_VBF_v4',
    'HLT_QuadPFJet82_65_48_35_BTagCSV_VBF_v4',
    'HLT_Random_v2',
    'HLT_RelIso1p0Mu20_v3',
    'HLT_RelIso1p0Mu5_v6',
    'HLT_RsqMR40_Rsq0p04_v6',
    'HLT_RsqMR55_Rsq0p09_MR150_v6',
    'HLT_RsqMR60_Rsq0p09_MR150_v6',
    'HLT_RsqMR65_Rsq0p09_MR150_v5',
    'HLT_SingleForJet15_v4',
    'HLT_SingleForJet25_v4',
    'HLT_SixJet35_v6',
    'HLT_SixJet45_v6',
    'HLT_SixJet50_v6',
    'HLT_Tau2Mu_ItTrack_v7',
    'HLT_TripleEle10_CaloIdL_TrkIdVL_v18',
    'HLT_TripleMu5_v19',
    'HLT_ZeroBiasPixel_DoubleTrack_v2',
    'HLT_ZeroBias_v7') ),
  ParkingMonitor = cms.vstring( 'HLT_BTagMu_Jet20_Mu4_v2',
    'HLT_BTagMu_Jet60_Mu4_v2',
    'HLT_DiJet35_MJJ650_AllJets_DEta3p5_VBF_v5',
    'HLT_DiJet35_MJJ700_AllJets_DEta3p5_VBF_v5',
    'HLT_DiJet35_MJJ750_AllJets_DEta3p5_VBF_v5',
    'HLT_Dimuon10_Jpsi_v6',
    'HLT_Dimuon5_PsiPrime_v6',
    'HLT_Dimuon5_Upsilon_v6',
    'HLT_Dimuon7_PsiPrime_v3',
    'HLT_Dimuon8_Jpsi_v7',
    'HLT_Dimuon8_Upsilon_v6',
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v3',
    'HLT_DoubleMu3p5_LowMassNonResonant_Displaced_v6',
    'HLT_DoubleMu3p5_LowMass_Displaced_v6',
    'HLT_HT200_AlphaT0p57_v8',
    'HLT_Mu13_Mu8_v21',
    'HLT_QuadJet50_Jet20_v4',
    'HLT_QuadJet50_v5',
    'HLT_RsqMR45_Rsq0p09_v5' ),
  PhotonHad = cms.vstring( 'HLT_Photon40_CaloIdL_RsqMR40_Rsq0p09_MR150_v6',
    'HLT_Photon40_CaloIdL_RsqMR45_Rsq0p09_MR150_v6',
    'HLT_Photon40_CaloIdL_RsqMR50_Rsq0p09_MR150_v6',
    'HLT_Photon60_CaloIdL_HT300_v4',
    'HLT_Photon60_CaloIdL_MHT70_v11',
    'HLT_Photon70_CaloIdXL_PFMET100_v6',
    'HLT_Photon70_CaloIdXL_PFNoPUHT400_v3',
    'HLT_Photon70_CaloIdXL_PFNoPUHT500_v3' ),
  RPCMonitor = cms.vstring( 'AlCa_RPCMuonNoHits_v9',
    'AlCa_RPCMuonNoTriggers_v9',
    'AlCa_RPCMuonNormalisation_v9' ),
  SingleElectron = cms.vstring( 'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_BTagIPIter_v8',
    'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_v7',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_DiCentralPFNoPUJet30_v1',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet30_v3',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet50_40_30_v3',
    'HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele27_WP80_CentralPFJet80_v8',
    'HLT_Ele27_WP80_PFMET_MT50_v6',
    'HLT_Ele27_WP80_WCandPt80_v8',
    'HLT_Ele27_WP80_v11',
    'HLT_Ele30_CaloIdVT_TrkIdT_v6',
    'HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11',
    'HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_PFMET20_v3',
    'HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_v3',
    'HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_CentralPFJet30_v3',
    'HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_v3',
    'HLT_Ele80_CaloIdVT_GsfTrkIdT_v2',
    'HLT_Ele90_CaloIdVT_GsfTrkIdT_v2' ),
  SingleMu = cms.vstring( 'HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3',
    'HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet30_v3',
    'HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_IsoMu20_WCandPt80_v3',
    'HLT_IsoMu20_eta2p1_CentralPFJet80_v8',
    'HLT_IsoMu20_eta2p1_v7',
    'HLT_IsoMu24_CentralPFJet30_CentralPFJet25_PFMET20_v3',
    'HLT_IsoMu24_CentralPFJet30_CentralPFJet25_v3',
    'HLT_IsoMu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3',
    'HLT_IsoMu24_eta2p1_v15',
    'HLT_IsoMu24_v17',
    'HLT_IsoMu30_eta2p1_v15',
    'HLT_IsoMu30_v11',
    'HLT_IsoMu34_eta2p1_v13',
    'HLT_IsoMu40_eta2p1_v10',
    'HLT_L2Mu70_2Cha_eta2p1_PFMET55_v1',
    'HLT_L2Mu70_2Cha_eta2p1_PFMET60_v1',
    'HLT_Mu12_eta2p1_DiCentral_20_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_BTagIP3D1stTrack_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_DiBTagIP3D1stTrack_v7',
    'HLT_Mu12_eta2p1_DiCentral_40_20_v7',
    'HLT_Mu12_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v6',
    'HLT_Mu12_v18',
    'HLT_Mu15_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v2',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_BTagIP3D1stTrack_v7',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_DiBTagIP3D1stTrack_v7',
    'HLT_Mu15_eta2p1_TriCentral_40_20_20_v7',
    'HLT_Mu15_eta2p1_v5',
    'HLT_Mu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3',
    'HLT_Mu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1',
    'HLT_Mu24_CentralPFJet30_CentralPFJet25_v3',
    'HLT_Mu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3',
    'HLT_Mu24_eta2p1_v5',
    'HLT_Mu24_v16',
    'HLT_Mu30_eta2p1_v5',
    'HLT_Mu30_v16',
    'HLT_Mu40_eta2p1_Track50_dEdx3p6_v5',
    'HLT_Mu40_eta2p1_Track60_dEdx3p7_v5',
    'HLT_Mu40_eta2p1_v11',
    'HLT_Mu40_v14',
    'HLT_Mu50_eta2p1_v8',
    'HLT_Mu5_v20',
    'HLT_RelIso1p0Mu20_v3',
    'HLT_RelIso1p0Mu5_v6' ),
  SinglePhoton = cms.vstring( 'HLT_DisplacedPhoton65EBOnly_CaloIdVL_IsoL_PFMET30_v3',
    'HLT_DisplacedPhoton65_CaloIdVL_IsoL_PFMET25_v3',
    'HLT_DoublePhoton5_IsoVL_CEP_v16',
    'HLT_L1DoubleEG3_FwdVeto_v2',
    'HLT_Photon135_v7',
    'HLT_Photon150_v4',
    'HLT_Photon160_v4',
    'HLT_Photon20_CaloIdVL_IsoL_v16',
    'HLT_Photon20_CaloIdVL_v4',
    'HLT_Photon300_NoHE_v5',
    'HLT_Photon30_CaloIdVL_v14',
    'HLT_Photon50_CaloIdVL_IsoL_v17',
    'HLT_Photon50_CaloIdVL_v10',
    'HLT_Photon75_CaloIdVL_v13',
    'HLT_Photon90_CaloIdVL_v10' ),
  Tau = cms.vstring( 'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v4',
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v3',
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v3',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_v9' ),
  TauParked = cms.vstring( 'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v4',
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v3',
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v3',
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v3',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v9',
    'HLT_LooseIsoPFTau35_Trk20_Prong1_v9' ),
  TauPlusX = cms.vstring( 'HLT_Ele22_eta2p1_WP90NoIso_LooseIsoPFTau20_v6',
    'HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v6',
    'HLT_IsoMu15_eta2p1_L1ETM20_v7',
    'HLT_IsoMu15_eta2p1_LooseIsoPFTau35_Trk20_Prong1_L1ETM20_v9',
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v6',
    'HLT_IsoMu18_eta2p1_MediumIsoPFTau25_Trk1_eta2p1_v3',
    'HLT_Mu15_eta2p1_L1ETM20_v5',
    'HLT_Mu17_eta2p1_LooseIsoPFTau20_v6' ),
  TestEnablesEcalHcalDT = cms.vstring( 'HLT_DTCalibration_v2',
    'HLT_EcalCalibration_v3',
    'HLT_HcalCalibration_v3' ),
  TestEnablesTracker = cms.vstring( 'HLT_TrackerCalibration_v3' ),
  VBF1Parked = cms.vstring( 'HLT_DiJet35_MJJ650_AllJets_DEta3p5_VBF_v5',
    'HLT_DiJet35_MJJ700_AllJets_DEta3p5_VBF_v5',
    'HLT_DiJet35_MJJ750_AllJets_DEta3p5_VBF_v5' )
)

GlobalTag = cms.ESSource( "PoolDBESSource",
  BlobStreamerName = cms.untracked.string( "TBufferBlobStreamingService" ),
  DBParameters = cms.PSet( 
    authenticationPath = cms.untracked.string( "." ),
    connectionRetrialTimeOut = cms.untracked.int32( 60 ),
    idleConnectionCleanupPeriod = cms.untracked.int32( 10 ),
    messageLevel = cms.untracked.int32( 0 ),
    enablePoolAutomaticCleanUp = cms.untracked.bool( False ),
    enableConnectionSharing = cms.untracked.bool( True ),
    enableReadOnlySessionOnUpdateConnection = cms.untracked.bool( False ),
    connectionTimeOut = cms.untracked.int32( 0 ),
    connectionRetrialPeriod = cms.untracked.int32( 10 )
  ),
  toGet = cms.VPSet( 
  ),
  connect = cms.string( "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG" ),
  globaltag = cms.string( "GR_H_V30::All" ),
  timetype = cms.string( "runnumber" ),
  RefreshEachRun = cms.untracked.bool( True )
)
HepPDTESSource = cms.ESSource( "HepPDTESSource",
  pdtFileName = cms.FileInPath( "SimGeneral/HepPDTESSource/data/pythiaparticle.tbl" )
)
eegeom = cms.ESSource( "EmptyESSource",
  iovIsRunNotTime = cms.bool( True ),
  recordName = cms.string( "EcalMappingRcd" ),
  firstValid = cms.vuint32( 1 )
)
es_hardcode = cms.ESSource( "HcalHardcodeCalibrations",
  fromDDD = cms.untracked.bool( False ),
  toGet = cms.untracked.vstring( 'GainWidths' )
)
hltESSBTagRecord = cms.ESSource( "EmptyESSource",
  iovIsRunNotTime = cms.bool( True ),
  recordName = cms.string( "JetTagComputerRecord" ),
  firstValid = cms.vuint32( 1 )
)
hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
  iovIsRunNotTime = cms.bool( True ),
  recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
  firstValid = cms.vuint32( 1 )
)
hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
  iovIsRunNotTime = cms.bool( True ),
  recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
  firstValid = cms.vuint32( 1 )
)
magfield = cms.ESSource( "XMLIdealGeometryESSource",
  geomXMLFiles = cms.vstring( 'Geometry/CMSCommonData/data/normal/cmsextent.xml',
    'Geometry/CMSCommonData/data/cms.xml',
    'Geometry/CMSCommonData/data/cmsMagneticField.xml',
    'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
    'MagneticField/GeomBuilder/data/MagneticFieldParameters_07_2pi.xml' ),
  rootNodeName = cms.string( "cmsMagneticField:MAGF" )
)

AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" )
)
AutoMagneticFieldESProducer = cms.ESProducer( "AutoMagneticFieldESProducer",
  label = cms.untracked.string( "" ),
  nominalCurrents = cms.untracked.vint32( -1, 0, 9558, 14416, 16819, 18268, 19262 ),
  valueOverride = cms.int32( -1 ),
  mapLabels = cms.untracked.vstring( '090322_3_8t',
    '0t',
    '071212_2t',
    '071212_3t',
    '071212_3_5t',
    '090322_3_8t',
    '071212_4t' )
)
CSCGeometryESModule = cms.ESProducer( "CSCGeometryESModule",
  useRealWireGeometry = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  alignmentsLabel = cms.string( "" ),
  useGangedStripsInME1a = cms.bool( True ),
  debugV = cms.untracked.bool( False ),
  useOnlyWiresInME1a = cms.bool( False ),
  useDDD = cms.bool( False ),
  useCentreTIOffsets = cms.bool( False ),
  applyAlignment = cms.bool( True )
)
CaloGeometryBuilder = cms.ESProducer( "CaloGeometryBuilder",
  SelectedCalos = cms.vstring( 'HCAL',
    'ZDC',
    'EcalBarrel',
    'EcalEndcap',
    'EcalPreshower',
    'TOWER' )
)
CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder" )
CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" )
)
CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  appendToDataLabel = cms.string( "" )
)
ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "ClusterShapeHitFilter" )
)
DTGeometryESModule = cms.ESProducer( "DTGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
EcalBarrelGeometryFromDBEP = cms.ESProducer( "EcalBarrelGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
EcalElectronicsMappingBuilder = cms.ESProducer( "EcalElectronicsMappingBuilder" )
EcalEndcapGeometryFromDBEP = cms.ESProducer( "EcalEndcapGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
EcalLaserCorrectionService = cms.ESProducer( "EcalLaserCorrectionService" )
EcalPreshowerGeometryFromDBEP = cms.ESProducer( "EcalPreshowerGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
EcalUnpackerWorkerESProducer = cms.ESProducer( "EcalUnpackerWorkerESProducer",
  CalibRHAlgo = cms.PSet( 
    flagsMapDBReco = cms.vint32( 0, 0, 0, 0, 4, -1, -1, -1, 4, 4, 7, 7, 7, 8, 9 ),
    Type = cms.string( "EcalRecHitWorkerSimple" ),
    killDeadChannels = cms.bool( True ),
    ChannelStatusToBeExcluded = cms.vint32( 10, 11, 12, 13, 14 ),
    laserCorrection = cms.bool( True ),
    EBLaserMIN = cms.double( 0.5 ),
    EELaserMIN = cms.double( 0.5 ),
    EBLaserMAX = cms.double( 2.0 ),
    EELaserMAX = cms.double( 3.0 )
  ),
  ComponentName = cms.string( "" ),
  UncalibRHAlgo = cms.PSet(  Type = cms.string( "EcalUncalibRecHitWorkerWeights" ) ),
  DCCDataUnpacker = cms.PSet( 
    orderedDCCIdList = cms.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 ),
    tccUnpacking = cms.bool( False ),
    srpUnpacking = cms.bool( False ),
    syncCheck = cms.bool( False ),
    feIdCheck = cms.bool( True ),
    headerUnpacking = cms.bool( True ),
    orderedFedList = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    feUnpacking = cms.bool( True ),
    forceKeepFRData = cms.bool( False ),
    memUnpacking = cms.bool( True )
  ),
  ElectronicsMapper = cms.PSet( 
    numbXtalTSamples = cms.uint32( 10 ),
    numbTriggerTSamples = cms.uint32( 1 )
  )
)
HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
HcalTopologyIdealEP = cms.ESProducer( "HcalTopologyIdealEP" )
MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
RPCGeometryESModule = cms.ESProducer( "RPCGeometryESModule",
  useDDD = cms.untracked.bool( False ),
  compatibiltyWith11 = cms.untracked.bool( True )
)
SiStripGainESProducer = cms.ESProducer( "SiStripGainESProducer",
  printDebug = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" ),
  APVGain = cms.VPSet( 
    cms.PSet(  Record = cms.string( "SiStripApvGainRcd" ),
      NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" )
    ),
    cms.PSet(  Record = cms.string( "SiStripApvGain2Rcd" ),
      NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" )
    )
  ),
  AutomaticNormalization = cms.bool( False )
)
SiStripQualityESProducer = cms.ESProducer( "SiStripQualityESProducer",
  appendToDataLabel = cms.string( "" ),
  PrintDebugOutput = cms.bool( False ),
  PrintDebug = cms.untracked.bool( False ),
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiStripDetVOffRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripDetCablingRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadChannelRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadFiberRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadModuleRcd" ),
      tag = cms.string( "" )
    )
  ),
  UseEmptyRunInfo = cms.bool( False ),
  ReduceGranularity = cms.bool( False ),
  ThresholdForReducedGranularity = cms.double( 0.3 )
)
SiStripRecHitMatcherESProducer = cms.ESProducer( "SiStripRecHitMatcherESProducer",
  ComponentName = cms.string( "StandardMatcher" ),
  NSigmaInside = cms.double( 3.0 )
)
SlaveField0 = cms.ESProducer( "UniformMagneticFieldESProducer",
  ZFieldInTesla = cms.double( 0.0 ),
  label = cms.untracked.string( "slave_0" )
)
SlaveField20 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "2_0T" ) ),
  label = cms.untracked.string( "slave_20" )
)
SlaveField30 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_0T" ) ),
  label = cms.untracked.string( "slave_30" )
)
SlaveField35 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_5T" ) ),
  label = cms.untracked.string( "slave_35" )
)
SlaveField38 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_8T" ) ),
  label = cms.untracked.string( "slave_38" )
)
SlaveField40 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "4_0T" ) ),
  label = cms.untracked.string( "slave_40" )
)
SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "SteppingHelixPropagatorAny" )
)
StripCPEfromTrackAngleESProducer = cms.ESProducer( "StripCPEESProducer",
  TanDiffusionAngle = cms.double( 0.01 ),
  UncertaintyScaling = cms.double( 1.42 ),
  ThicknessRelativeUncertainty = cms.double( 0.02 ),
  MaybeNoiseThreshold = cms.double( 3.5 ),
  ComponentName = cms.string( "StripCPEfromTrackAngle" ),
  MinimumUncertainty = cms.double( 0.01 ),
  NoiseThreshold = cms.double( 2.3 )
)
TrackerDigiGeometryESModule = cms.ESProducer( "TrackerDigiGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
  fromDDD = cms.bool( False )
)
TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" )
)
VBF0 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "0t" ),
  version = cms.string( "grid_1103l_071212_2t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_0" ),
  cacheLastVolume = cms.untracked.bool( True )
)
VBF20 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_2t" ),
  version = cms.string( "grid_1103l_071212_2t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_20" ),
  cacheLastVolume = cms.untracked.bool( True )
)
VBF30 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_3t" ),
  version = cms.string( "grid_1103l_071212_3t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_30" ),
  cacheLastVolume = cms.untracked.bool( True )
)
VBF35 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_3_5t" ),
  version = cms.string( "grid_1103l_071212_3_5t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_35" ),
  cacheLastVolume = cms.untracked.bool( True )
)
VBF38 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32( 14100, 14200, 17600, 17800, 17900, 18100, 18300, 18400, 18600, 23100, 23300, 23400, 23600, 23800, 23900, 24100, 28600, 28800, 28900, 29100, 29300, 29400, 29600, 28609, 28809, 28909, 29109, 29309, 29409, 29609, 28610, 28810, 28910, 29110, 29310, 29410, 29610, 28611, 28811, 28911, 29111, 29311, 29411, 29611 ),
  overrideMasterSector = cms.bool( False ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble( 1.0, 1.0, 0.994, 1.004, 1.004, 1.005, 1.004, 1.004, 0.994, 0.965, 0.958, 0.958, 0.953, 0.958, 0.958, 0.965, 0.918, 0.924, 0.924, 0.906, 0.924, 0.924, 0.918, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991 ),
  label = cms.untracked.string( "090322_3_8t" ),
  version = cms.string( "grid_1103l_090322_3_8t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_38" ),
  cacheLastVolume = cms.untracked.bool( True )
)
VBF40 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_4t" ),
  version = cms.string( "grid_1103l_071212_4t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_40" ),
  cacheLastVolume = cms.untracked.bool( True )
)
ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" )
)
ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  includeBadChambers = cms.bool( False )
)
ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
  dbstatusMask = cms.PSet( 
    kGood = cms.vuint32( 0 ),
    kProblematic = cms.vuint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ),
    kRecovered = cms.vuint32(  ),
    kTime = cms.vuint32(  ),
    kWeird = cms.vuint32(  ),
    kBad = cms.vuint32( 11, 12, 13, 14, 15, 16 )
  ),
  timeThresh = cms.double( 2.0 ),
  flagMask = cms.PSet( 
    kGood = cms.vstring( 'kGood' ),
    kProblematic = cms.vstring( 'kPoorReco',
      'kPoorCalib',
      'kNoisy',
      'kSaturated' ),
    kRecovered = cms.vstring( 'kLeadingEdgeRecovered',
      'kTowerRecovered' ),
    kTime = cms.vstring( 'kOutOfTime' ),
    kWeird = cms.vstring( 'kWeird',
      'kDiWeird' ),
    kBad = cms.vstring( 'kFaultyHardware',
      'kDead',
      'kKilled' )
  )
)
hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  RecoveredRecHitBits = cms.vstring( 'TimingAddedBit',
    'TimingSubtractedBit' ),
  SeverityLevels = cms.VPSet( 
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 0 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerProb' ),
      Level = cms.int32( 1 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HSCP_R1R2',
  'HSCP_FracLeader',
  'HSCP_OuterEnergy',
  'HSCP_ExpFit',
  'ADCSaturationBit' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 5 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HBHEHpdHitMultiplicity',
  'HFDigiTime',
  'HBHEPulseShape',
  'HOBit',
  'HFInTimeWindow',
  'ZDCBit',
  'CalibrationBit',
  'TimingErrorBit',
  'HBHEFlatNoise',
  'HBHESpikeNoise',
  'HBHETriangleNoise',
  'HBHETS4TS5Noise' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 8 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HFLongShort',
  'HFS8S1Ratio',
  'HFPET' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 11 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerMask' ),
      Level = cms.int32( 12 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellHot' ),
      Level = cms.int32( 15 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellOff',
        'HcalCellDead' ),
      Level = cms.int32( 20 )
    )
  ),
  DropChannelStatusBits = cms.vstring( 'HcalCellMask',
    'HcalCellOff',
    'HcalCellDead' )
)
hcal_db_producer = cms.ESProducer( "HcalDbProducer" )
hltCombinedSecondaryVertex = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  categoryVariableName = cms.string( "vertexCategory" ),
  useTrackWeights = cms.bool( True ),
  useCategories = cms.bool( True ),
  pseudoMultiplicityMin = cms.uint32( 2 ),
  correctVertexMass = cms.bool( True ),
  trackSelection = cms.PSet( 
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    maxDistToAxis = cms.double( 0.07 ),
    sip2dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( -99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  calibrationRecords = cms.vstring( 'CombinedSVRecoVertex',
    'CombinedSVPseudoVertex',
    'CombinedSVNoVertex' ),
  trackPairV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.03 ) ),
  charmCut = cms.double( 1.5 ),
  vertexFlip = cms.bool( False ),
  minimumTrackWeight = cms.double( 0.5 ),
  pseudoVertexV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
  trackMultiplicityMin = cms.uint32( 3 ),
  trackPseudoSelection = cms.PSet( 
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    maxDistToAxis = cms.double( 0.07 ),
    sip2dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( 2.0 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  trackSort = cms.string( "sip2dSig" ),
  trackFlip = cms.bool( False )
)
hltESPAK5CaloL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1FastJetCorrectionESProducer',
    'hltESPL2RelativeCorrectionESProducer',
    'hltESPL3AbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
hltESPAK5CaloL2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL2RelativeCorrectionESProducer',
    'hltESPL3AbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
hltESPAK5PFL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1PFFastJetCorrectionESProducer',
    'hltESPL2PFRelativeCorrectionESProducer',
    'hltESPL3PFAbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
hltESPAK5PFNoPUL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1PFNoPUFastJetCorrectionESProducer',
    'hltESPL2PFNoPURelativeCorrectionESProducer',
    'hltESPL3PFNoPUAbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
hltESPBwdAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPBwdAnalyticalPropagator" ),
  PropagationDirection = cms.string( "oppositeToMomentum" )
)
hltESPBwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPBwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
hltESPChi2EstimatorForRefit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2EstimatorForRefit" )
)
hltESPChi2MeasurementEstimator = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator" )
)
hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" )
)
hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" )
)
hltESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkf3HitTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPCkf3HitTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPCkf3HitTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.9 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPCkf3HitTrajectoryFilter" )
)
hltESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPCkfTrajectoryBuilderForHI = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilterForHI" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPCkfTrajectoryBuilderForHI" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTrackerForHI" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.9 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPCkfTrajectoryFilter" )
)
hltESPCkfTrajectoryFilterForHI = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minimumNumberOfHits = cms.int32( 6 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 11.0 )
  ),
  ComponentName = cms.string( "hltESPCkfTrajectoryFilterForHI" )
)
hltESPCloseComponentsMerger5D = cms.ESProducer( "CloseComponentsMergerESProducer5D",
  ComponentName = cms.string( "hltESPCloseComponentsMerger5D" ),
  MaxComponents = cms.int32( 12 ),
  DistanceMeasure = cms.string( "hltESPKullbackLeiblerDistance5D" )
)
hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPESUnpackerWorker = cms.ESProducer( "ESUnpackerWorkerESProducer",
  RHAlgo = cms.PSet( 
    ESRecoAlgo = cms.int32( 0 ),
    Type = cms.string( "ESRecHitWorker" )
  ),
  DCCDataUnpacker = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) ),
  ComponentName = cms.string( "hltESPESUnpackerWorker" )
)
hltESPEcalRegionCablingESProducer = cms.ESProducer( "EcalRegionCablingESProducer",
  esMapping = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) )
)
hltESPEcalTrigTowerConstituentsMapBuilder = cms.ESProducer( "EcalTrigTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/EcalMapping/data/EndCap_TTMap.txt" )
)
hltESPElectronChi2 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 2000.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPElectronChi2" )
)
hltESPElectronMaterialEffects = cms.ESProducer( "GsfMaterialEffectsESProducer",
  BetheHeitlerParametrization = cms.string( "BetheHeitler_cdfmom_nC6_O5.par" ),
  EnergyLossUpdator = cms.string( "GsfBetheHeitlerUpdator" ),
  ComponentName = cms.string( "hltESPElectronMaterialEffects" ),
  MultipleScatteringUpdator = cms.string( "MultipleScatteringUpdator" ),
  Mass = cms.double( 5.11E-4 ),
  BetheHeitlerCorrection = cms.int32( 2 )
)
hltESPFastSteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorAny" )
)
hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" )
)
hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 10.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPFittingSmootherIT" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPFittingSmootherRK" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
hltESPFwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPFwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" )
)
hltESPGlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )
hltESPGsfElectronFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPGsfTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPGsfTrajectorySmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPGsfElectronFittingSmoother" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
hltESPGsfTrajectoryFitter = cms.ESProducer( "GsfTrajectoryFitterESProducer",
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectoryFitter" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  GeometricalPropagator = cms.string( "hltESPAnalyticalPropagator" )
)
hltESPGsfTrajectorySmoother = cms.ESProducer( "GsfTrajectorySmootherESProducer",
  ErrorRescaling = cms.double( 100.0 ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectorySmoother" ),
  GeometricalPropagator = cms.string( "hltESPBwdAnalyticalPropagator" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" )
)
hltESPHIMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
    'FPix2_pos+TEC1_pos',
    'FPix2_pos+TEC2_pos',
    'TEC1_pos+TEC2_pos',
    'TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC1_neg',
    'FPix2_neg+TEC2_neg',
    'TEC1_neg+TEC2_neg',
    'TEC2_neg+TEC3_neg' ),
  ComponentName = cms.string( "hltESPHIMixedLayerPairs" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    useErrorsFromParam = cms.bool( True )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.006 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    useErrorsFromParam = cms.bool( True )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPHIPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPHIPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPHIPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPHIPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPHITTRHBuilderWithoutRefit = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "Fake" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "Fake" ),
  ComponentName = cms.string( "hltESPHITTRHBuilderWithoutRefit" )
)
hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPKFTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmoother" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 20.0 ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  Fitter = cms.string( "hltESPRKFitter" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPRKSmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" )
)
hltESPKullbackLeiblerDistance5D = cms.ESProducer( "DistanceBetweenComponentsESProducer5D",
  ComponentName = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  DistanceMeasure = cms.string( "KullbackLeibler" )
)
hltESPL1FastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6CaloJets','rho' ),
  algorithm = cms.string( "AK5Calo" ),
  level = cms.string( "L1FastJet" )
)
hltESPL1PFFastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6PFJets','rho' ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L1FastJet" )
)
hltESPL1PFNoPUFastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6PFJets','rho' ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L1FastJet" )
)
hltESPL2PFNoPURelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L2Relative" )
)
hltESPL2PFRelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L2Relative" )
)
hltESPL2RelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5Calo" ),
  level = cms.string( "L2Relative" )
)
hltESPL3AbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5Calo" ),
  level = cms.string( "L3Absolute" )
)
hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPL3PFAbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L3Absolute" )
)
hltESPL3PFNoPUAbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L3Absolute" )
)
hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltESPMeasurementTrackerForHI = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( False ),
  Regional = cms.bool( False ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripRawToDigi' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltHISiStripClustersNonRegional" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    ),
    TID = cms.PSet( 
      maxBad = cms.uint32( 4 ),
      maxConsecutiveBad = cms.uint32( 2 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTrackerForHI" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltESPMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
    'FPix2_pos+TEC1_pos',
    'FPix2_pos+TEC2_pos',
    'TEC1_pos+TEC2_pos',
    'TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC1_neg',
    'FPix2_neg+TEC2_neg',
    'TEC1_neg+TEC2_neg',
    'TEC2_neg+TEC3_neg' ),
  ComponentName = cms.string( "hltESPMixedLayerPairs" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPMuTrackJpsiEffTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuTrackJpsiEffTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltESPMuTrackJpsiEffTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPMuTrackJpsiEffTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 9 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPMuTrackJpsiEffTrajectoryFilter" )
)
hltESPMuTrackJpsiTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuTrackJpsiTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltESPMuTrackJpsiTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPMuTrackJpsiTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPMuTrackJpsiTrajectoryFilter" )
)
hltESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( False ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPMuonCkfTrajectoryBuilderSeedHit = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryBuilderSeedHit" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( True ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPMuonCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.9 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 )
  ),
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryFilter" )
)
hltESPMuonDetLayerGeometryESProducer = cms.ESProducer( "MuonDetLayerGeometryESProducer" )
hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
)
hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
  EdgeClusterErrorX = cms.double( 50.0 ),
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  UseErrorsFromTemplates = cms.bool( True ),
  eff_charge_cut_highX = cms.double( 1.0 ),
  TruncatePixelCharge = cms.bool( True ),
  size_cutY = cms.double( 3.0 ),
  size_cutX = cms.double( 3.0 ),
  inflate_all_errors_no_trk_angle = cms.bool( False ),
  IrradiationBiasCorrection = cms.bool( False ),
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  inflate_errors = cms.bool( False ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True )
)
hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False )
)
hltESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPixelLayerTripletsHITHB = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3' ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsHITHB" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPixelLayerTripletsHITHE = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsHITHE" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPixelLayerTripletsReg = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsReg" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    useErrorsFromParam = cms.bool( True ),
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.006 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    useErrorsFromParam = cms.bool( True ),
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
  maxImpactParameterSig = cms.double( 999999.0 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.03 ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  nthTrack = cms.int32( -1 )
)
hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKSmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)
hltESPSiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagator" )
)
hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAny" )
)
hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" )
)
hltESPSmartPropagatorOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorOpposite" )
)
hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 )
)
hltESPSoftLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer",
  ipSign = cms.string( "any" )
)
hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorAlong" )
)
hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorOpposite" )
)
hltESPStraightLinePropagator = cms.ESProducer( "StraightLinePropagatorESProducer",
  ComponentName = cms.string( "hltESPStraightLinePropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" )
)
hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" )
)
hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" )
)
hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
)
hltESPTrackCounting3D1st = cms.ESProducer( "TrackCountingESProducer",
  deltaR = cms.double( -1.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  maximumDecayLength = cms.double( 5.0 ),
  nthTrack = cms.int32( 1 )
)
hltESPTrackCounting3D2nd = cms.ESProducer( "TrackCountingESProducer",
  deltaR = cms.double( -1.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  maximumDecayLength = cms.double( 5.0 ),
  nthTrack = cms.int32( 2 )
)
hltESPTrackerRecoGeometryESProducer = cms.ESProducer( "TrackerRecoGeometryESProducer",
  appendToDataLabel = cms.string( "" ),
  trackerGeometryLabel = cms.untracked.string( "" )
)
hltESPTrajectoryBuilderForElectrons = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "hltESPFwdElectronPropagator" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterForElectrons" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderForElectrons" ),
  propagatorOpposite = cms.string( "hltESPBwdElectronPropagator" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPElectronChi2" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 90.0 )
)
hltESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPTrajectoryBuilderL3 = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterL3" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderL3" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( False )
)
hltESPTrajectoryCleanerBySharedSeeds = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedSeeds" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( True )
)
hltESPTrajectoryFilterForElectrons = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    minPt = cms.double( 2.0 ),
    minHitsMinPt = cms.int32( -1 ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPTrajectoryFilterForElectrons" )
)
hltESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPTrajectoryFilterIT" )
)
hltESPTrajectoryFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.5 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 1000000000 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPTrajectoryFilterL3" )
)
hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPbJetRegionalTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPbJetRegionalTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltESPbJetRegionalTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltESPbJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPbJetRegionalTrajectoryFilter" )
)
hltHIAllESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkf3HitTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPCkf3HitTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltHIAllESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltHIAllESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltHISiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltHISiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltHIAllESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltHIAllESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( False ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
hltHIAllESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltIter1ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter1SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter1ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter1ESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1ESPPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltIter1ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltIter1ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.2 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryFilterIT" )
)
hltIter1Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter1Tau3MuSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter1Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter1Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter1Tau3MuESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1Tau3MuESPPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltIter1Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltIter2ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter2SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter2ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter2ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter2ESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltIter2ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltIter2ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryFilterIT" )
)
hltIter2Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter2SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter2Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter2Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter2Tau3MuESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter2Tau3MuESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltIter2Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltIter3ESPLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg',
    'BPix2+FPix1_pos+FPix2_pos',
    'BPix2+FPix1_neg+FPix2_neg',
    'FPix1_pos+FPix2_pos+TEC1_pos',
    'FPix1_neg+FPix2_neg+TEC1_neg',
    'FPix2_pos+TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC2_neg+TEC3_neg',
    'BPix2+BPix3+TIB1',
    'BPix2+BPix3+TIB2',
    'BPix1+BPix3+TIB1',
    'BPix1+BPix3+TIB2',
    'BPix1+BPix2+TIB1',
    'BPix1+BPix2+TIB2' ),
  ComponentName = cms.string( "hltIter3ESPLayerTriplets" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
hltIter3ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter3SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter3ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter3ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltIter3ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 0 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryFilterIT" )
)
hltIter3Tau3MuESPLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg',
    'BPix2+FPix1_pos+FPix2_pos',
    'BPix2+FPix1_neg+FPix2_neg',
    'FPix1_pos+FPix2_pos+TEC1_pos',
    'FPix1_neg+FPix2_neg+TEC1_neg',
    'FPix2_pos+TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC2_neg+TEC3_neg',
    'BPix2+BPix3+TIB1',
    'BPix2+BPix3+TIB2',
    'BPix1+BPix3+TIB1',
    'BPix1+BPix3+TIB2',
    'BPix1+BPix2+TIB1',
    'BPix1+BPix2+TIB2' ),
  ComponentName = cms.string( "hltIter3Tau3MuESPLayerTriplets" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
hltIter3Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter3Tau3MuSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter3Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter3Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter3Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
hltIter4ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter4SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter4ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter4ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter4ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'TIB1+TIB2' ),
  ComponentName = cms.string( "hltIter4ESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet(  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet(  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
hltIter4ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter4ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  minNrOfHitsForRebuild = cms.untracked.int32( 4 )
)
hltIter4ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 0 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 6 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltIter4ESPTrajectoryFilterIT" )
)
hltIter4Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter4Tau3MuSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  badStripCuts = cms.PSet( 
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter4Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter4Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
hltIter4Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter4Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  minNrOfHitsForRebuild = cms.untracked.int32( 4 )
)
hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  includeBadChambers = cms.bool( False )
)
navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" )
)
preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  includeBadChambers = cms.bool( False )
)
siPixelQualityESProducer = cms.ESProducer( "SiPixelQualityESProducer",
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiPixelQualityFromDbRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiPixelDetVOffRcd" ),
      tag = cms.string( "" )
    )
  )
)
siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer" )
siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
  LatencyRecord = cms.PSet( 
    record = cms.string( "SiStripLatencyRcd" ),
    label = cms.untracked.string( "" )
  ),
  LorentzAngleDeconvMode = cms.PSet( 
    record = cms.string( "SiStripLorentzAngleRcd" ),
    label = cms.untracked.string( "deconvolution" )
  ),
  LorentzAnglePeakMode = cms.PSet( 
    record = cms.string( "SiStripLorentzAngleRcd" ),
    label = cms.untracked.string( "peak" )
  )
)
sistripconn = cms.ESProducer( "SiStripConnectivity" )

FastTimerService = cms.Service( "FastTimerService",
  dqmPath = cms.untracked.string( "HLT/TimerService" ),
  skipFirstPath = cms.untracked.bool( False ),
  dqmModuleTimeRange = cms.untracked.double( 40.0 ),
  enableDQMbyPathCounters = cms.untracked.bool( False ),
  enableDQMbyLumi = cms.untracked.bool( False ),
  dqmTimeResolution = cms.untracked.double( 5.0 ),
  enableTimingPaths = cms.untracked.bool( True ),
  dqmModuleTimeResolution = cms.untracked.double( 0.2 ),
  dqmPathTimeResolution = cms.untracked.double( 0.5 ),
  useRealTimeClock = cms.untracked.bool( True ),
  enableTimingModules = cms.untracked.bool( True ),
  dqmPathTimeRange = cms.untracked.double( 100.0 ),
  enableDQM = cms.untracked.bool( True ),
  enableDQMbyPathDetails = cms.untracked.bool( False ),
  dqmTimeRange = cms.untracked.double( 1000.0 ),
  enableDQMbyPathOverhead = cms.untracked.bool( False ),
  enableDQMbyModule = cms.untracked.bool( False ),
  enableDQMbyPathActive = cms.untracked.bool( False ),
  enableTimingSummary = cms.untracked.bool( False ),
  enableDQMbyPathTotal = cms.untracked.bool( True )
)
DQM = cms.Service( "DQM",
  publishFrequency = cms.untracked.double( 5.0 ),
  debug = cms.untracked.bool( False ),
  collectorPort = cms.untracked.int32( 0 ),
  collectorHost = cms.untracked.string( "" )
)
DQMStore = cms.Service( "DQMStore",
)
DTDataIntegrityTask = cms.Service( "DTDataIntegrityTask",
  processingMode = cms.untracked.string( "HLT" ),
  fedIntegrityFolder = cms.untracked.string( "DT/FEDIntegrity_EvF" ),
  getSCInfo = cms.untracked.bool( True )
)
FUShmDQMOutputService = cms.Service( "FUShmDQMOutputService",
  lumiSectionInterval = cms.untracked.int32( 0 ),
  compressionLevel = cms.int32( 1 ),
  initialMessageBufferSize = cms.untracked.int32( 1000000 ),
  lumiSectionsPerUpdate = cms.double( 1.0 ),
  useCompression = cms.bool( True )
)
MessageLogger = cms.Service( "MessageLogger",
  suppressInfo = cms.untracked.vstring(  ),
  debugs = cms.untracked.PSet( 
    threshold = cms.untracked.string( "INFO" ),
    placeholder = cms.untracked.bool( True ),
    suppressInfo = cms.untracked.vstring(  ),
    suppressWarning = cms.untracked.vstring(  ),
    suppressDebug = cms.untracked.vstring(  ),
    suppressError = cms.untracked.vstring(  )
  ),
  suppressDebug = cms.untracked.vstring(  ),
  cout = cms.untracked.PSet( 
    threshold = cms.untracked.string( "ERROR" ),
    suppressInfo = cms.untracked.vstring(  ),
    suppressWarning = cms.untracked.vstring(  ),
    suppressDebug = cms.untracked.vstring(  ),
    suppressError = cms.untracked.vstring(  )
  ),
  cerr_stats = cms.untracked.PSet( 
    threshold = cms.untracked.string( "WARNING" ),
    output = cms.untracked.string( "cerr" ),
    optionalPSet = cms.untracked.bool( True )
  ),
  warnings = cms.untracked.PSet( 
    threshold = cms.untracked.string( "INFO" ),
    placeholder = cms.untracked.bool( True ),
    suppressInfo = cms.untracked.vstring(  ),
    suppressWarning = cms.untracked.vstring(  ),
    suppressDebug = cms.untracked.vstring(  ),
    suppressError = cms.untracked.vstring(  )
  ),
  statistics = cms.untracked.vstring( 'cerr' ),
  cerr = cms.untracked.PSet( 
    INFO = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
    noTimeStamps = cms.untracked.bool( False ),
    FwkReport = cms.untracked.PSet( 
      reportEvery = cms.untracked.int32( 1 ),
      limit = cms.untracked.int32( 0 )
    ),
    default = cms.untracked.PSet(  limit = cms.untracked.int32( 10000000 ) ),
    Root_NoDictionary = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
    FwkJob = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
    FwkSummary = cms.untracked.PSet( 
      reportEvery = cms.untracked.int32( 1 ),
      limit = cms.untracked.int32( 10000000 )
    ),
    threshold = cms.untracked.string( "INFO" ),
    suppressInfo = cms.untracked.vstring(  ),
    suppressWarning = cms.untracked.vstring(  ),
    suppressDebug = cms.untracked.vstring(  ),
    suppressError = cms.untracked.vstring(  )
  ),
  FrameworkJobReport = cms.untracked.PSet( 
    default = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
    FwkJob = cms.untracked.PSet(  limit = cms.untracked.int32( 10000000 ) )
  ),
  suppressWarning = cms.untracked.vstring( 'hltOnlineBeamSpot',
    'hltCtf3HitL1SeededWithMaterialTracks',
    'hltL3MuonsOIState',
    'hltPixelTracksForHighMult',
    'hltHITPixelTracksHE',
    'hltHITPixelTracksHB',
    'hltCtfL1SeededWithMaterialTracks',
    'hltRegionalTracksForL3MuonIsolation',
    'hltSiPixelClusters',
    'hltActivityStartUpElectronPixelSeeds',
    'hltLightPFTracks',
    'hltPixelVertices3DbbPhi',
    'hltL3MuonsIOHit',
    'hltPixelTracks',
    'hltSiPixelDigis',
    'hltL3MuonsOIHit',
    'hltL1SeededElectronGsfTracks',
    'hltL1SeededStartUpElectronPixelSeeds',
    'hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJetFastPV',
    'hltCtfActivityWithMaterialTracks' ),
  errors = cms.untracked.PSet( 
    threshold = cms.untracked.string( "INFO" ),
    placeholder = cms.untracked.bool( True ),
    suppressInfo = cms.untracked.vstring(  ),
    suppressWarning = cms.untracked.vstring(  ),
    suppressDebug = cms.untracked.vstring(  ),
    suppressError = cms.untracked.vstring(  )
  ),
  fwkJobReports = cms.untracked.vstring( 'FrameworkJobReport' ),
  debugModules = cms.untracked.vstring(  ),
  infos = cms.untracked.PSet( 
    threshold = cms.untracked.string( "INFO" ),
    Root_NoDictionary = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
    placeholder = cms.untracked.bool( True ),
    suppressInfo = cms.untracked.vstring(  ),
    suppressWarning = cms.untracked.vstring(  ),
    suppressDebug = cms.untracked.vstring(  ),
    suppressError = cms.untracked.vstring(  )
  ),
  categories = cms.untracked.vstring( 'FwkJob',
    'FwkReport',
    'FwkSummary',
    'Root_NoDictionary' ),
  destinations = cms.untracked.vstring( 'warnings',
    'errors',
    'infos',
    'debugs',
    'cout',
    'cerr' ),
  threshold = cms.untracked.string( "INFO" ),
  suppressError = cms.untracked.vstring( 'hltOnlineBeamSpot',
    'hltL3MuonCandidates',
    'hltL3TkTracksFromL2OIState',
    'hltPFJetCtfWithMaterialTracks',
    'hltL3TkTracksFromL2IOHit',
    'hltL3TkTracksFromL2OIHit' )
)
MicroStateService = cms.Service( "MicroStateService",
)
ModuleWebRegistry = cms.Service( "ModuleWebRegistry",
)
PrescaleService = cms.Service( "PrescaleService",
  forceDefault = cms.bool( False ),
  prescaleTable = cms.VPSet(  *(
    cms.PSet(  pathName = cms.string( "HLT_Activity_Ecal_SC7_v13" ),
      prescales = cms.vuint32( 280, 280, 280, 280, 280, 280, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1SingleJet16_v7" ),
      prescales = cms.vuint32( 55, 55, 55, 70, 70, 70, 200, 200, 450, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1SingleJet36_v7" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 300, 100, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFJet40_v7" ),
      prescales = cms.vuint32( 5, 5, 5, 2, 2, 2, 27, 27, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFJet80_v8" ),
      prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 15, 15, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFJet140_v8" ),
      prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 10, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFJet200_v8" ),
      prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFJet260_v8" ),
      prescales = cms.vuint32( 30, 30, 30, 20, 20, 20, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFJet320_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Jet370_NoJetID_v15" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFJet400_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MonoCentralPFJet80_PFMETnoMu105_NHEF0p95_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_SingleForJet25_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_SingleForJet15_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJetAve40_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 120, 100, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJetAve80_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 8, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJetAve140_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 6, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJetAve200_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJetAve260_v9" ),
      prescales = cms.vuint32( 15, 15, 15, 10, 10, 10, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJetAve320_v9" ),
      prescales = cms.vuint32( 5, 5, 5, 4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJetAve400_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_FatDiPFJetMass750_DR1p1_Deta1p5_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleJet20_ForwardBackward_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiJet80_DiJet60_DiJet20_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJet40_PFMETnoMu65_MJJ800VBF_AllJets_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJet40_PFMETnoMu65_MJJ600VBF_LeadingJets_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiJet40Eta2p6_BTagIP3DFastPV_v7" ),
      prescales = cms.vuint32( 5, 5, 5, 4, 4, 4, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiJet80Eta2p6_BTagIP3DFastPVLoose_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Jet60Eta1p7_Jet53Eta1p7_DiBTagIP3DFastPV_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Jet80Eta1p7_Jet70Eta1p7_DiBTagIP3DFastPV_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Jet160Eta2p4_Jet120Eta2p4_DiBTagIP3DFastPVLoose_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet50_v5" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet50_Jet20_v4" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet60_DiJet20_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet70_v6" ),
      prescales = cms.vuint32( 35, 35, 35, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet80_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet90_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet75_55_35_20_BTagIP_VBF_v6" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadJet75_55_38_20_BTagIP_VBF_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadPFJet78_61_44_31_BTagCSV_VBF_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_QuadPFJet82_65_48_35_BTagCSV_VBF_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_SixJet35_v6" ),
      prescales = cms.vuint32( 15, 15, 15, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_SixJet45_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_SixJet50_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_EightJet30_eta3p0_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_EightJet35_eta3p0_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_ExclDiJet35_HFOR_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_ExclDiJet35_HFAND_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_ExclDiJet80_HFAND_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 8, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_JetE30_NoBPTX_v14" ),
      prescales = cms.vuint32( 160, 160, 60, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_JetE30_NoBPTX3BX_NoHalo_v16" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_JetE50_NoBPTX3BX_NoHalo_v13" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_JetE70_NoBPTX3BX_NoHalo_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT200_AlphaT0p57_v8" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT200_v6" ),
      prescales = cms.vuint32( 4800, 4800, 4800, 4800, 4000, 4000, 4000, 4000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT250_AlphaT0p55_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT250_AlphaT0p57_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT250_v7" ),
      prescales = cms.vuint32( 2400, 2400, 2400, 2400, 2000, 2000, 2000, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT300_AlphaT0p53_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT300_AlphaT0p54_v14" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT300_v7" ),
      prescales = cms.vuint32( 1200, 1200, 1200, 1200, 1000, 1000, 1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT300_DoubleDisplacedPFJet60_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT300_DoubleDisplacedPFJet60_ChgFraction10_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT300_SingleDisplacedPFJet60_v9" ),
      prescales = cms.vuint32( 150, 150, 150, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT300_SingleDisplacedPFJet60_ChgFraction10_v9" ),
      prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT350_v7" ),
      prescales = cms.vuint32( 600, 600, 600, 600, 500, 500, 500, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT350_AlphaT0p52_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT350_AlphaT0p53_v19" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT400_v7" ),
      prescales = cms.vuint32( 300, 300, 300, 300, 250, 250, 250, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT400_AlphaT0p51_v19" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT400_AlphaT0p52_v14" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT450_AlphaT0p51_v14" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT450_v7" ),
      prescales = cms.vuint32( 150, 150, 150, 150, 125, 125, 125, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT500_v7" ),
      prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT550_v7" ),
      prescales = cms.vuint32( 70, 70, 70, 60, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT650_v7" ),
      prescales = cms.vuint32( 25, 25, 25, 25, 25, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT650_Track50_dEdx3p6_v10" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT650_Track60_dEdx3p7_v10" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HT750_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT350_v3" ),
      prescales = cms.vuint32( 1000, 1000, 1000, 1000, 1000, 1000, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT650_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT650_DiCentralPFNoPUJet80_CenPFNoPUJet40_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT700_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT750_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFMET150_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFMET180_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiCentralJetSumpT100_dPhi05_DiCentralPFJet60_25_PFMET100_HBHENoiseCleaned_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiCentralPFJet30_PFMET80_v5" ),
      prescales = cms.vuint32( 150, 150, 60, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiCentralPFNoPUJet50_PFMETORPFMETNoMu80_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiCentralPFJet30_PFMET80_BTagCSV07_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d03_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05d05_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiPFJet80_DiPFJet30_BTagCSVd07d05_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET80_v5" ),
      prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET80_Track50_dEdx3p6_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET80_Track60_dEdx3p7_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET120_v12" ),
      prescales = cms.vuint32( 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET120_HBHENoiseCleaned_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET200_v12" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET200_HBHENoiseCleaned_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET300_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET300_HBHENoiseCleaned_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET400_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_MET400_HBHENoiseCleaned_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1SingleMuOpen_v7" ),
      prescales = cms.vuint32( 30, 30, 25, 20, 15, 10, 30, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1SingleMu12_v2" ),
      prescales = cms.vuint32( 25, 25, 25, 17, 17, 17, 150, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2Mu70_2Cha_eta2p1_PFMET55_v1" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2Mu70_2Cha_eta2p1_PFMET60_v1" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2Mu20_eta2p1_NoVertex_v2" ),
      prescales = cms.vuint32( 5000, 5000, 5000, 5000, 5000, 5000, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo_v4" ),
      prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2Mu20_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2Mu30_NoVertex_2Cha_NoBPTX3BX_NoHalo_v1" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2TripleMu10_0_0_NoVertex_PFJet40Neutral_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleDisplacedMu4_DiPFJet40Neutral_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu5_v20" ),
      prescales = cms.vuint32( 30000, 30000, 30000, 30000, 30000, 30000, 24000, 24000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_v18" ),
      prescales = cms.vuint32( 4, 4, 4, 4, 4, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu12_v18" ),
      prescales = cms.vuint32( 30, 30, 30, 30, 30, 30, 600, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu17_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu12_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v6" ),
      prescales = cms.vuint32( 1000, 1000, 1000, 1000, 1000, 1000, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu15_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v2" ),
      prescales = cms.vuint32( 1000, 1000, 1000, 1000, 1000, 1000, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu15_eta2p1_v5" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 800, 800, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu24_v16" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu24_eta2p1_v5" ),
      prescales = cms.vuint32( 300, 300, 300, 250, 250, 250, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu30_v16" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu30_eta2p1_v5" ),
      prescales = cms.vuint32( 300, 300, 300, 250, 250, 250, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu40_v14" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu40_eta2p1_v11" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu50_eta2p1_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RelIso1p0Mu5_v6" ),
      prescales = cms.vuint32( 20, 20, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RelIso1p0Mu20_v3" ),
      prescales = cms.vuint32( 550, 550, 550, 500, 500, 500, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu15_eta2p1_L1ETM20_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu20_eta2p1_v7" ),
      prescales = cms.vuint32( 300, 300, 300, 250, 250, 250, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu24_v17" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu24_eta2p1_v15" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu30_v11" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu30_eta2p1_v15" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu34_eta2p1_v13" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu40_eta2p1_v10" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu40_eta2p1_Track50_dEdx3p6_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu40_eta2p1_Track60_dEdx3p7_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2DoubleMu23_NoVertex_v11" ),
      prescales = cms.vuint32( 20, 20, 20, 20, 20, 20, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2DoubleMu23_NoVertex_2Cha_Angle2p5_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L2DoubleMu38_NoVertex_2Cha_Angle2p5_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu11_Acoplanarity03_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu4_Jpsi_Displaced_v12" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu4_JpsiTk_Displaced_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu3_4_Dimuon5_Bs_Central_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu3p5_4_Dimuon5_Bs_Central_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu4_Dimuon7_Bs_Forward_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu3p5_LowMass_Displaced_v6" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu3p5_LowMassNonResonant_Displaced_v6" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon0_Jpsi_v17" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon0_Jpsi_NoVertexing_v14" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 8, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon0_Upsilon_v17" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon0_PsiPrime_v6" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon5_Upsilon_v6" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon5_PsiPrime_v6" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon7_Upsilon_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon7_PsiPrime_v3" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon8_Jpsi_v7" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon8_Upsilon_v6" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon10_Jpsi_v6" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon11_Upsilon_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon0_Jpsi_Muon_v18" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon0_Upsilon_Muon_v18" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Dimuon3p5_SameSign_v6" ),
      prescales = cms.vuint32( 7, 7, 7, 6, 6, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu4_Acoplanarity03_v5" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Tau2Mu_ItTrack_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu13_Mu8_v21" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu17_Mu8_v21" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu17_TkMu8_v13" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu22_TkMu8_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu22_TkMu22_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_TripleMu5_v19" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu5_IsoMu5_v20" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu5_L2Mu3_Jpsi_v7" ),
      prescales = cms.vuint32( 150, 150, 150, 100, 100, 100, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu5_Track2_Jpsi_v21" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu5_Track3p5_Jpsi_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu7_Track7_Jpsi_v20" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BTagMu_Jet20_Mu4_v2" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BTagMu_Jet60_Mu4_v2" ),
      prescales = cms.vuint32( 0, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon20_CaloIdVL_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon20_CaloIdVL_IsoL_v16" ),
      prescales = cms.vuint32( 17, 17, 17, 20, 30, 15000, 800, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon22_R9Id90_HE10_Iso40_EBOnly_v5" ),
      prescales = cms.vuint32( 270, 270, 270, 210, 210, 210, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_Photon18_v12" ),
      prescales = cms.vuint32( 1600, 1600, 1600, 1600, 1600, 1600, 65, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_R9Id85_Photon18_R9Id85_Mass60_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_R9Id85_Photon18_CaloId10_Iso50_Mass60_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_CaloId10_Iso50_Photon18_R9Id85_Mass60_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_CaloId10_Iso50_Photon18_CaloId10_Iso50_Mass60_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass60_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass70_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_v5" ),
      prescales = cms.vuint32( 1400, 1400, 1400, 1200, 1200, 1200, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon30_CaloIdVL_v14" ),
      prescales = cms.vuint32( 7000, 7000, 7000, 7000, 7000, 7000, 280, 280, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_R9Id90_HE10_Iso40_EBOnly_v5" ),
      prescales = cms.vuint32( 90, 90, 90, 70, 70, 70, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_Photon22_v6" ),
      prescales = cms.vuint32( 800, 800, 800, 800, 800, 800, 20, 20, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_R9Id85_Photon22_R9Id85_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_R9Id85_Photon22_CaloId10_Iso50_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_CaloId10_Iso50_Photon22_R9Id85_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_CaloId10_Iso50_Photon22_CaloId10_Iso50_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_v5" ),
      prescales = cms.vuint32( 300, 300, 300, 300, 300, 300, 10, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon50_CaloIdVL_v10" ),
      prescales = cms.vuint32( 900, 900, 900, 900, 900, 900, 35, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon50_CaloIdVL_IsoL_v17" ),
      prescales = cms.vuint32( 330, 330, 330, 330, 330, 330, 10, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon50_R9Id90_HE10_Iso40_EBOnly_v5" ),
      prescales = cms.vuint32( 30, 30, 30, 30, 30, 30, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon60_CaloIdL_MHT70_v11" ),
      prescales = cms.vuint32( 20, 20, 20, 20, 20, 20, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon60_CaloIdL_HT300_v4" ),
      prescales = cms.vuint32( 60, 60, 60, 60, 60, 60, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon70_CaloIdXL_PFNoPUHT400_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon70_CaloIdXL_PFNoPUHT500_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon70_CaloIdXL_PFMET100_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon75_CaloIdVL_v13" ),
      prescales = cms.vuint32( 150, 150, 150, 150, 150, 150, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon75_R9Id90_HE10_Iso40_EBOnly_v5" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon90_CaloIdVL_v10" ),
      prescales = cms.vuint32( 100, 100, 100, 80, 60, 60, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon90_R9Id90_HE10_Iso40_EBOnly_v5" ),
      prescales = cms.vuint32( 5, 5, 5, 5, 5, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DisplacedPhoton65_CaloIdVL_IsoL_PFMET25_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DisplacedPhoton65EBOnly_CaloIdVL_IsoL_PFMET30_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon135_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon150_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon160_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon300_NoHE_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoublePhoton48_HEVT_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoublePhoton53_HEVT_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoublePhoton70_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoublePhoton80_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoublePhoton5_IsoVL_CEP_v16" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1SingleEG5_v6" ),
      prescales = cms.vuint32( 1800, 1800, 1800, 1800, 14400, 36000, 1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1SingleEG12_v6" ),
      prescales = cms.vuint32( 34, 34, 34, 40, 60, 30000, 350, 125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1DoubleEG3_FwdVeto_v2" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1ETM30_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1500, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1ETM40_v2" ),
      prescales = cms.vuint32( 1750, 1750, 1750, 1500, 1500, 1500, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1ETM70_v2" ),
      prescales = cms.vuint32( 150, 150, 150, 150, 150, 150, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1ETM100_v2" ),
      prescales = cms.vuint32( 95, 95, 95, 85, 85, 85, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_TrkIdVL_v5" ),
      prescales = cms.vuint32( 40, 40, 40, 40, 320, 800, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_TrkIdVL_EG7_v2" ),
      prescales = cms.vuint32( 40, 40, 40, 40, 400, 800, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_TrkIdVL_Jet30_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 15, 100, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdL_CaloIsoVL_v17" ),
      prescales = cms.vuint32( 40, 40, 40, 40, 320, 800, 1600, 1600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v15" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 160, 400, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele17_CaloIdL_CaloIsoVL_v17" ),
      prescales = cms.vuint32( 17, 17, 17, 20, 30, 15000, 900, 900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 750, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v18" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass50_v6" ),
      prescales = cms.vuint32( 5, 5, 5, 5, 5, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v6" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele22_CaloIdL_CaloIsoVL_v6" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11" ),
      prescales = cms.vuint32( 300, 300, 300, 300, 300, 300, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele15_CaloIdT_CaloIsoVL_trackless_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele27_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT15_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele23_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_HFT30_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet30_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet45_35_25_v1" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_TriCentralPFNoPUJet50_40_30_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele27_WP80_v11" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele27_WP80_PFMET_MT50_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele30_CaloIdVT_TrkIdT_v6" ),
      prescales = cms.vuint32( 300, 300, 300, 250, 250, 250, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v11" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_Mass50_v6" ),
      prescales = cms.vuint32( 7, 7, 7, 6, 6, 6, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele80_CaloIdVT_GsfTrkIdT_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele90_CaloIdVT_GsfTrkIdT_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle8_CaloIdT_TrkIdVL_v12" ),
      prescales = cms.vuint32( 2, 2, 2, 2, 32, 80, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle33_CaloIdL_v14" ),
      prescales = cms.vuint32( 40, 40, 40, 40, 40, 40, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle33_CaloIdT_v10" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_LooseIsoPFTau35_Trk20_Prong1_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu15_eta2p1_LooseIsoPFTau35_Trk20_Prong1_L1ETM20_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v3" ),
      prescales = cms.vuint32( 70, 70, 70, 60, 50, 50, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu18_eta2p1_MediumIsoPFTau25_Trk1_eta2p1_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BTagMu_DiJet20_Mu5_v6" ),
      prescales = cms.vuint32( 7, 7, 7, 6, 6, 6, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BTagMu_DiJet40_Mu5_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BTagMu_DiJet70_Mu5_v6" ),
      prescales = cms.vuint32( 8, 8, 8, 7, 7, 7, 70, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BTagMu_DiJet110_Mu5_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BTagMu_Jet300_Mu5_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu7_Ele7_CaloIdT_CaloIsoVL_v7" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_DiJet30_v7" ),
      prescales = cms.vuint32( 20, 20, 20, 20, 20, 20, 160, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_TriJet30_v7" ),
      prescales = cms.vuint32( 3, 3, 3, 3, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_QuadJet30_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu12_DoubleCentralJet65_v4" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu15_eta2p1_L1ETM20_v5" ),
      prescales = cms.vuint32( 85, 85, 85, 70, 70, 70, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3" ),
      prescales = cms.vuint32( 150, 150, 60, 10, 10, 10, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu24_CentralPFJet30_CentralPFJet25_v3" ),
      prescales = cms.vuint32( 15, 15, 15, 10, 10, 10, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu24_CentralPFJet30_CentralPFJet25_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu24_CentralPFJet30_CentralPFJet25_PFMET20_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_v3" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele32_WP80_PFJet30_PFJet25_Deta3_CentralPFJet30_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele32_WP80_CentralPFJet35_CentralPFJet25_PFMET20_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_v3" ),
      prescales = cms.vuint32( 20, 20, 20, 20, 20, 20, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_v3" ),
      prescales = cms.vuint32( 16, 16, 16, 16, 16, 16, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet30_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v3" ),
      prescales = cms.vuint32( 96, 96, 96, 96, 96, 96, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu17_eta2p1_TriCentralPFNoPUJet45_35_25_v1" ),
      prescales = cms.vuint32( 32, 32, 32, 32, 32, 32, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu20_WCandPt80_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu12_eta2p1_DiCentral_40_20_DiBTagIP3D1stTrack_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu12_eta2p1_DiCentral_40_20_BTagIP3D1stTrack_v7" ),
      prescales = cms.vuint32( 160, 160, 160, 160, 130, 130, 130, 130, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu12_eta2p1_DiCentral_40_20_v7" ),
      prescales = cms.vuint32( 570, 570, 570, 570, 470, 470, 470, 470, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu12_eta2p1_DiCentral_20_v7" ),
      prescales = cms.vuint32( 1050, 1050, 1050, 1050, 870, 870, 870, 870, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu15_eta2p1_TriCentral_40_20_20_DiBTagIP3D1stTrack_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu15_eta2p1_TriCentral_40_20_20_BTagIP3D1stTrack_v7" ),
      prescales = cms.vuint32( 70, 70, 70, 70, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu15_eta2p1_TriCentral_40_20_20_v7" ),
      prescales = cms.vuint32( 180, 180, 180, 180, 150, 150, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu30_Ele30_CaloIdL_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_PFNoPUHT350_PFMHT40_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu20_eta2p1_CentralPFJet80_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT175_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleRelIso1p0Mu5_Mass8_PFNoPUHT225_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu8_Mass8_PFNoPUHT175_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu8_Mass8_PFNoPUHT225_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RelIso1p0Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT350_Mu15_PFMET45_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT350_Mu15_PFMET50_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT400_Mu5_PFMET45_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT400_Mu5_PFMET50_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu40_PFNoPUHT350_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu60_PFNoPUHT350_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v16" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu8_Ele8_CaloIdT_TrkIdVL_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 160, 400, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v18" ),
      prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v18" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v18" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Jet30_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 750, 160, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_v7" ),
      prescales = cms.vuint32( 125, 125, 125, 100, 75, 25, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele25_CaloIdVT_CaloIsoVL_TrkIdVL_TrkIsoT_DiCentralPFNoPUJet30_v1" ),
      prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_BTagIPIter_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele27_WP80_CentralPFJet80_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele27_WP80_WCandPt80_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet100_PFNoPUJet25_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele30_CaloIdVT_TrkIdT_PFNoPUJet150_PFNoPUJet25_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT175_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_PFNoPUHT225_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v12" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_TripleEle10_CaloIdL_TrkIdVL_v18" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RsqMR40_Rsq0p04_v6" ),
      prescales = cms.vuint32( 150, 150, 150, 150, 150, 150, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RsqMR45_Rsq0p09_v5" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RsqMR55_Rsq0p09_MR150_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RsqMR60_Rsq0p09_MR150_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_RsqMR65_Rsq0p09_MR150_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu12_RsqMR30_Rsq0p04_MR200_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu12_RsqMR40_Rsq0p04_MR200_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR30_Rsq0p04_MR200_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_RsqMR40_Rsq0p04_MR200_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele12_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_DoubleCentralJet65_v4" ),
      prescales = cms.vuint32( 200, 200, 200, 100, 100, 100, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon40_CaloIdL_RsqMR40_Rsq0p09_MR150_v6" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon40_CaloIdL_RsqMR45_Rsq0p09_MR150_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Photon40_CaloIdL_RsqMR50_Rsq0p09_MR150_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoublePhoton40_CaloIdL_Rsq0p035_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoublePhoton40_CaloIdL_Rsq0p06_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu22_Photon22_CaloIdL_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_DoubleEle8_CaloIdT_TrkIdVL_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu14_Mass8_PFMET40_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleMu14_Mass8_PFMET50_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DoubleEle14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET40_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu14_Ele14_CaloIdT_TrkIdVL_Mass8_PFMET50_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT350_PFMET100_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PFNoPUHT400_PFMET100_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_CleanPFNoPUHT300_Ele40_CaloIdVT_TrkIdT_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_CleanPFNoPUHT300_Ele60_CaloIdVT_TrkIdT_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele5_SC5_Jpsi_Mass2to15_v4" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiJet35_MJJ650_AllJets_DEta3p5_VBF_v5" ),
      prescales = cms.vuint32( 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiJet35_MJJ700_AllJets_DEta3p5_VBF_v5" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DiJet35_MJJ750_AllJets_DEta3p5_VBF_v5" ),
      prescales = cms.vuint32( 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Ele22_eta2p1_WP90NoIso_LooseIsoPFTau20_v6" ),
      prescales = cms.vuint32( 55, 55, 55, 50, 40, 40, 40, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Mu17_eta2p1_LooseIsoPFTau20_v6" ),
      prescales = cms.vuint32( 64, 64, 64, 55, 45, 45, 45, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PixelTracks_Multiplicity70_v3" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PixelTracks_Multiplicity80_v12" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_PixelTracks_Multiplicity90_v3" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "DST_HT250_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "DST_L1HTT_Or_L1MultiJet_v4" ),
      prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "DST_Mu5_HT250_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "DST_Ele8_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT250_v4" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BeamGas_HF_Beam1_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BeamGas_HF_Beam2_v5" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_BeamHalo_v13" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoTrackHE_v15" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_IsoTrackHB_v14" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HcalPhiSym_v11" ),
      prescales = cms.vuint32( 15, 15, 15, 15, 15, 15, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HcalNZS_v10" ),
      prescales = cms.vuint32( 15, 15, 15, 15, 15, 15, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_GlobalRunHPDNoise_v8" ),
      prescales = cms.vuint32( 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 40 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1Tech_HBHEHO_totalOR_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1Tech_HCAL_HF_single_channel_v4" ),
      prescales = cms.vuint32( 500, 500, 500, 500, 500, 500, 500, 500, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_ZeroBias_v7" ),
      prescales = cms.vuint32( 150, 150, 150, 150, 150, 150, 500, 500, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_ZeroBiasPixel_DoubleTrack_v2" ),
      prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Physics_v5" ),
      prescales = cms.vuint32( 8000, 8000, 8000, 8000, 8000, 8000, 8000, 3000, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80 )
    ),
    cms.PSet(  pathName = cms.string( "DST_Physics_v5" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DTCalibration_v2" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_EcalCalibration_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_HcalCalibration_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_TrackerCalibration_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_Random_v2" ),
      prescales = cms.vuint32( 200, 200, 200, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 10000 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1SingleMuOpen_AntiBPTX_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1TrackerCosmics_v7" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_DTErrors_v3" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_L1DoubleJet36Central_v7" ),
      prescales = cms.vuint32( 730, 730, 730, 730, 730, 730, 4800, 4800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_EcalPi0EBonly_v6" ),
      prescales = cms.vuint32( 3, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_EcalPi0EEonly_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_EcalEtaEBonly_v6" ),
      prescales = cms.vuint32( 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_EcalEtaEEonly_v6" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_EcalPhiSym_v13" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_RPCMuonNoTriggers_v9" ),
      prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_RPCMuonNoHits_v9" ),
      prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_RPCMuonNormalisation_v9" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_LumiPixels_v8" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_LumiPixels_ZeroBias_v4" ),
      prescales = cms.vuint32( 18, 18, 18, 18, 18, 18, 62, 62, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "AlCa_LumiPixels_Random_v1" ),
      prescales = cms.vuint32( 30, 30, 30, 30, 30, 30, 30, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "DQM_FEDIntegrity_v11" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 )
    ),
    cms.PSet(  pathName = cms.string( "HLT_LogMonitor_v4" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "AOutput" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "BOutput" ),
      prescales = cms.vuint32( 20, 20, 20, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "ALCALUMIPIXELSOutput" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
    ),
    cms.PSet(  pathName = cms.string( "DQMOutput" ),
      prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "ExpressOutput" ),
      prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
    ),
    cms.PSet(  pathName = cms.string( "HLTMONOutput" ),
      prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 )
    )
) ),
  lvl1DefaultLabel = cms.string( "3e33" ),
  lvl1Labels = cms.vstring( '8e33nopark',
    '8e33',
    '7e33',
    '6e33',
    '4e33',
    '2e33',
    '72_bunches',
    '28_bunches',
    '5e32',
    '6000Hz',
    '5000Hz',
    '4000Hz',
    '3000Hz',
    '2000Hz',
    '1500Hz',
    '1000Hz',
    '500Hz',
    'EM1',
    'EM2',
    'CirculatingBeam',
    'CirculatingBeam+HighRandom' )
)
UpdaterService = cms.Service( "UpdaterService",
)

