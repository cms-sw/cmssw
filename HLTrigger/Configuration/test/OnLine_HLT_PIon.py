# /dev/CMSSW_5_2_6/PIon/V109 (CMSSW_5_2_8)

import FWCore.ParameterSet.Config as cms

process = cms.Process( "HLTPIon" )

process.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_5_2_6/PIon/V109')
)

process.streams = cms.PSet( 
  A = cms.vstring( 'Commissioning',
    'Cosmics',
    'HcalHPDNoise',
    'HcalNZS',
    'LogMonitor',
    'MinimumBias',
    'PAHighPt',
    'PAMinBiasUPC',
    'PAMuon',
    'PPFSQ',
    'PPJet',
    'PPMinBias',
    'PPMuon',
    'PPPhoton',
    'SingleElectron',
    'SingleMu' ),
  ALCALUMIPIXELS = cms.vstring( 'AlCaLumiPixels' ),
  ALCAP0 = cms.vstring( 'AlCaP0' ),
  ALCAPHISYM = cms.vstring( 'AlCaPhiSym' ),
  Calibration = cms.vstring( 'TestEnablesEcalHcalDT' ),
  DQM = cms.vstring( 'OnlineMonitor' ),
  EcalCalibration = cms.vstring( 'EcalLaser' ),
  Express = cms.vstring( 'ExpressPhysics' ),
  HLTDQM = cms.vstring( 'OnlineHltMonitor' ),
  NanoDST = cms.vstring( 'L1Accept' ),
  RPCMON = cms.vstring( 'RPCMonitor' ),
  TrackerCalibration = cms.vstring( 'TestEnablesTracker' )
)
process.datasets = cms.PSet( 
  AlCaLumiPixels = cms.vstring( 'AlCa_LumiPixels_Random_v1',
    'AlCa_LumiPixels_ZeroBias_v4',
    'AlCa_LumiPixels_v8' ),
  AlCaP0 = cms.vstring( 'AlCa_PAEcalEtaEBonly_v1',
    'AlCa_PAEcalEtaEEonly_v1',
    'AlCa_PAEcalPi0EBonly_v1',
    'AlCa_PAEcalPi0EEonly_v1' ),
  AlCaPhiSym = cms.vstring( 'AlCa_EcalPhiSym_v13' ),
  Commissioning = cms.vstring( 'HLT_Activity_Ecal_SC7_v13',
    'HLT_BeamGas_HF_Beam1_v5',
    'HLT_BeamGas_HF_Beam2_v5' ),
  Cosmics = cms.vstring( 'HLT_BeamHalo_v13',
    'HLT_L1SingleMuOpen_AntiBPTX_v7',
    'HLT_L1TrackerCosmics_v7' ),
  EcalLaser = cms.vstring( 'HLT_EcalCalibration_v3' ),
  ExpressPhysics = cms.vstring( 'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_Mu15_eta2p1_v5',
    'HLT_PABTagMu_Jet20_Mu4_v1',
    'HLT_PABptxMinusNotBptxPlus_v1',
    'HLT_PABptxPlusNotBptxMinus_v1',
    'HLT_PACastorEmNotHfCoincidencePm_v1',
    'HLT_PACastorEmNotHfSingleChannel_v1',
    'HLT_PACastorEmTotemLowMultiplicity_v1',
    'HLT_PADimuon0_NoVertexing_v1',
    'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
    'HLT_PADoubleJet20_ForwardBackward_v1',
    'HLT_PADoubleMu4_Acoplanarity03_v1',
    'HLT_PAExclDijet35_HFAND_v1',
    'HLT_PAExclDijet35_HFOR_v1',
    'HLT_PAForJet100Eta2_v1',
    'HLT_PAForJet100Eta3_v1',
    'HLT_PAForJet20Eta2_v1',
    'HLT_PAForJet20Eta3_v1',
    'HLT_PAForJet40Eta2_v1',
    'HLT_PAForJet40Eta3_v1',
    'HLT_PAForJet60Eta2_v1',
    'HLT_PAForJet60Eta3_v1',
    'HLT_PAForJet80Eta2_v1',
    'HLT_PAForJet80Eta3_v1',
    'HLT_PAFullTrack12_v2',
    'HLT_PAFullTrack20_v2',
    'HLT_PAFullTrack30_v2',
    'HLT_PAFullTrack50_v2',
    'HLT_PAHFOR_SingleTrack_v1',
    'HLT_PAHFSumET100_v3',
    'HLT_PAHFSumET140_v3',
    'HLT_PAHFSumET170_v3',
    'HLT_PAHFSumET210_v3',
    'HLT_PAJet100_NoJetID_v1',
    'HLT_PAJet120_NoJetID_v1',
    'HLT_PAJet20_NoJetID_v1',
    'HLT_PAJet40ETM30_v1',
    'HLT_PAJet40_NoJetID_v1',
    'HLT_PAJet60ETM30_v1',
    'HLT_PAJet60_NoJetID_v1',
    'HLT_PAJet80_NoJetID_v1',
    'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
    'HLT_PAL1DoubleEG3_FwdVeto_v1',
    'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
    'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
    'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
    'HLT_PAL1DoubleMu0_HighQ_v1',
    'HLT_PAL1DoubleMu0_v1',
    'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
    'HLT_PAL1DoubleMuOpen_v1',
    'HLT_PAL1SingleEG20_TotemDiffractive_v1',
    'HLT_PAL1SingleJet16_v1',
    'HLT_PAL1SingleJet36_v1',
    'HLT_PAL1SingleJet52_TotemDiffractive_v1',
    'HLT_PAL1SingleMu20_TotemDiffractive_v1',
    'HLT_PAL1Tech53_MB_SingleTrack_v1',
    'HLT_PAL1Tech53_MB_v1',
    'HLT_PAL1Tech54_ZeroBias_v1',
    'HLT_PAL1Tech63_CASTORHaloMuon_v1',
    'HLT_PAL1Tech_HBHEHO_totalOR_v1',
    'HLT_PAL2DoubleMu3_v1',
    'HLT_PAMinBiasBHC_OR_v1',
    'HLT_PAMinBiasBHC_v1',
    'HLT_PAMinBiasHF_OR_v1',
    'HLT_PAMinBiasHF_v1',
    'HLT_PAMinBiasHfOrBHC_v1',
    'HLT_PAMu12_v1',
    'HLT_PAMu3PFJet20_v1',
    'HLT_PAMu3PFJet40_v1',
    'HLT_PAMu3_v1',
    'HLT_PAMu7PFJet20_v1',
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
    'HLT_PAMu7_v1',
    'HLT_PAPhoton10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton20_TightCaloIdVL_v1',
    'HLT_PAPhoton30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton30_TightCaloIdVL_v1',
    'HLT_PAPhoton40_NoCaloIdVL_v1',
    'HLT_PAPhoton40_TightCaloIdVL_v1',
    'HLT_PAPhoton60_NoCaloIdVL_v1',
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
    'HLT_PAPixelTracks_Multiplicity100_v2',
    'HLT_PAPixelTracks_Multiplicity130_v2',
    'HLT_PAPixelTracks_Multiplicity160_v2',
    'HLT_PAPixelTracks_Multiplicity190_v2',
    'HLT_PAPixelTracks_Multiplicity220_v2',
    'HLT_PARandom_v1',
    'HLT_PARomanPots_Tech52_v1',
    'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
    'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
    'HLT_PASingleForJet15_v1',
    'HLT_PASingleForJet25_v1',
    'HLT_PAT1minbias_Tech55_v1',
    'HLT_PATech35_HFSumET100_v3',
    'HLT_PATech35_v1',
    'HLT_PATripleJet100_20_20_v1',
    'HLT_PATripleJet20_20_20_v1',
    'HLT_PATripleJet40_20_20_v1',
    'HLT_PATripleJet60_20_20_v1',
    'HLT_PATripleJet80_20_20_v1',
    'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
    'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
    'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
    'HLT_PAZeroBiasPixel_DoubleTrack_v1',
    'HLT_PAZeroBiasPixel_SingleTrack_v1',
    'HLT_PAZeroBias_v1',
    'HLT_PPL1DoubleJetC36_v1',
    'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
    'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
    'HLT_PPPixelTracks_Multiplicity55_v1',
    'HLT_PPPixelTracks_Multiplicity70_v1',
    'HLT_PPPixelTracks_Multiplicity85_v1' ),
  HcalHPDNoise = cms.vstring( 'HLT_GlobalRunHPDNoise_v8' ),
  HcalNZS = cms.vstring( 'HLT_PAHcalNZS_v1',
    'HLT_PAHcalPhiSym_v1',
    'HLT_PAHcalUTCA_v1' ),
  L1Accept = cms.vstring( 'DST_Physics_v5' ),
  LogMonitor = cms.vstring( 'HLT_LogMonitor_v4' ),
  MinimumBias = cms.vstring( 'HLT_Physics_v5' ),
  OnlineHltMonitor = cms.vstring( 'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_Mu15_eta2p1_v5',
    'HLT_PABTagMu_Jet20_Mu4_v1',
    'HLT_PABptxMinusNotBptxPlus_v1',
    'HLT_PABptxPlusNotBptxMinus_v1',
    'HLT_PACastorEmNotHfCoincidencePm_v1',
    'HLT_PACastorEmNotHfSingleChannel_v1',
    'HLT_PACastorEmTotemLowMultiplicity_v1',
    'HLT_PADimuon0_NoVertexing_v1',
    'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
    'HLT_PADoubleJet20_ForwardBackward_v1',
    'HLT_PADoubleMu4_Acoplanarity03_v1',
    'HLT_PAExclDijet35_HFAND_v1',
    'HLT_PAExclDijet35_HFOR_v1',
    'HLT_PAForJet100Eta2_v1',
    'HLT_PAForJet100Eta3_v1',
    'HLT_PAForJet20Eta2_v1',
    'HLT_PAForJet20Eta3_v1',
    'HLT_PAForJet40Eta2_v1',
    'HLT_PAForJet40Eta3_v1',
    'HLT_PAForJet60Eta2_v1',
    'HLT_PAForJet60Eta3_v1',
    'HLT_PAForJet80Eta2_v1',
    'HLT_PAForJet80Eta3_v1',
    'HLT_PAFullTrack12_v2',
    'HLT_PAFullTrack20_v2',
    'HLT_PAFullTrack30_v2',
    'HLT_PAFullTrack50_v2',
    'HLT_PAHFOR_SingleTrack_v1',
    'HLT_PAHFSumET100_v3',
    'HLT_PAHFSumET140_v3',
    'HLT_PAHFSumET170_v3',
    'HLT_PAHFSumET210_v3',
    'HLT_PAHcalNZS_v1',
    'HLT_PAHcalPhiSym_v1',
    'HLT_PAHcalUTCA_v1',
    'HLT_PAJet100_NoJetID_v1',
    'HLT_PAJet120_NoJetID_v1',
    'HLT_PAJet20_NoJetID_v1',
    'HLT_PAJet40ETM30_v1',
    'HLT_PAJet40_NoJetID_v1',
    'HLT_PAJet60ETM30_v1',
    'HLT_PAJet60_NoJetID_v1',
    'HLT_PAJet80_NoJetID_v1',
    'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
    'HLT_PAL1DoubleEG3_FwdVeto_v1',
    'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
    'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
    'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
    'HLT_PAL1DoubleMu0_HighQ_v1',
    'HLT_PAL1DoubleMu0_v1',
    'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
    'HLT_PAL1DoubleMuOpen_v1',
    'HLT_PAL1SingleEG20_TotemDiffractive_v1',
    'HLT_PAL1SingleJet16_v1',
    'HLT_PAL1SingleJet36_v1',
    'HLT_PAL1SingleJet52_TotemDiffractive_v1',
    'HLT_PAL1SingleMu20_TotemDiffractive_v1',
    'HLT_PAL1Tech53_MB_SingleTrack_v1',
    'HLT_PAL1Tech53_MB_v1',
    'HLT_PAL1Tech54_ZeroBias_v1',
    'HLT_PAL1Tech63_CASTORHaloMuon_v1',
    'HLT_PAL1Tech_HBHEHO_totalOR_v1',
    'HLT_PAL2DoubleMu3_v1',
    'HLT_PAMinBiasBHC_OR_v1',
    'HLT_PAMinBiasBHC_v1',
    'HLT_PAMinBiasHF_OR_v1',
    'HLT_PAMinBiasHF_v1',
    'HLT_PAMinBiasHfOrBHC_v1',
    'HLT_PAMu12_v1',
    'HLT_PAMu3PFJet20_v1',
    'HLT_PAMu3PFJet40_v1',
    'HLT_PAMu3_v1',
    'HLT_PAMu7PFJet20_v1',
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
    'HLT_PAMu7_v1',
    'HLT_PAPhoton10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton20_TightCaloIdVL_v1',
    'HLT_PAPhoton30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton30_TightCaloIdVL_v1',
    'HLT_PAPhoton40_NoCaloIdVL_v1',
    'HLT_PAPhoton40_TightCaloIdVL_v1',
    'HLT_PAPhoton60_NoCaloIdVL_v1',
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
    'HLT_PAPixelTracks_Multiplicity100_v2',
    'HLT_PAPixelTracks_Multiplicity130_v2',
    'HLT_PAPixelTracks_Multiplicity160_v2',
    'HLT_PAPixelTracks_Multiplicity190_v2',
    'HLT_PAPixelTracks_Multiplicity220_v2',
    'HLT_PARandom_v1',
    'HLT_PARomanPots_Tech52_v1',
    'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
    'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
    'HLT_PASingleForJet15_v1',
    'HLT_PASingleForJet25_v1',
    'HLT_PAT1minbias_Tech55_v1',
    'HLT_PATech35_HFSumET100_v3',
    'HLT_PATech35_v1',
    'HLT_PATripleJet100_20_20_v1',
    'HLT_PATripleJet20_20_20_v1',
    'HLT_PATripleJet40_20_20_v1',
    'HLT_PATripleJet60_20_20_v1',
    'HLT_PATripleJet80_20_20_v1',
    'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
    'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
    'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
    'HLT_PAZeroBiasPixel_DoubleTrack_v1',
    'HLT_PAZeroBiasPixel_SingleTrack_v1',
    'HLT_PAZeroBias_v1',
    'HLT_PPL1DoubleJetC36_v1',
    'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
    'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
    'HLT_PPPixelTracks_Multiplicity55_v1',
    'HLT_PPPixelTracks_Multiplicity70_v1',
    'HLT_PPPixelTracks_Multiplicity85_v1' ),
  OnlineMonitor = cms.vstring( 'HLT_Activity_Ecal_SC7_v13',
    'HLT_BeamGas_HF_Beam1_v5',
    'HLT_BeamGas_HF_Beam2_v5',
    'HLT_BeamHalo_v13',
    'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_GlobalRunHPDNoise_v8',
    'HLT_L1SingleMuOpen_AntiBPTX_v7',
    'HLT_L1TrackerCosmics_v7',
    'HLT_Mu15_eta2p1_v5',
    'HLT_PABTagMu_Jet20_Mu4_v1',
    'HLT_PABptxMinusNotBptxPlus_v1',
    'HLT_PABptxPlusNotBptxMinus_v1',
    'HLT_PACastorEmNotHfCoincidencePm_v1',
    'HLT_PACastorEmNotHfSingleChannel_v1',
    'HLT_PACastorEmTotemLowMultiplicity_v1',
    'HLT_PADimuon0_NoVertexing_v1',
    'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
    'HLT_PADoubleJet20_ForwardBackward_v1',
    'HLT_PADoubleMu4_Acoplanarity03_v1',
    'HLT_PAExclDijet35_HFAND_v1',
    'HLT_PAExclDijet35_HFOR_v1',
    'HLT_PAForJet100Eta2_v1',
    'HLT_PAForJet100Eta3_v1',
    'HLT_PAForJet20Eta2_v1',
    'HLT_PAForJet20Eta3_v1',
    'HLT_PAForJet40Eta2_v1',
    'HLT_PAForJet40Eta3_v1',
    'HLT_PAForJet60Eta2_v1',
    'HLT_PAForJet60Eta3_v1',
    'HLT_PAForJet80Eta2_v1',
    'HLT_PAForJet80Eta3_v1',
    'HLT_PAFullTrack12_v2',
    'HLT_PAFullTrack20_v2',
    'HLT_PAFullTrack30_v2',
    'HLT_PAFullTrack50_v2',
    'HLT_PAHFOR_SingleTrack_v1',
    'HLT_PAHFSumET100_v3',
    'HLT_PAHFSumET140_v3',
    'HLT_PAHFSumET170_v3',
    'HLT_PAHFSumET210_v3',
    'HLT_PAHcalNZS_v1',
    'HLT_PAHcalPhiSym_v1',
    'HLT_PAHcalUTCA_v1',
    'HLT_PAJet100_NoJetID_v1',
    'HLT_PAJet120_NoJetID_v1',
    'HLT_PAJet20_NoJetID_v1',
    'HLT_PAJet40ETM30_v1',
    'HLT_PAJet40_NoJetID_v1',
    'HLT_PAJet60ETM30_v1',
    'HLT_PAJet60_NoJetID_v1',
    'HLT_PAJet80_NoJetID_v1',
    'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
    'HLT_PAL1DoubleEG3_FwdVeto_v1',
    'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
    'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
    'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
    'HLT_PAL1DoubleMu0_HighQ_v1',
    'HLT_PAL1DoubleMu0_v1',
    'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
    'HLT_PAL1DoubleMuOpen_v1',
    'HLT_PAL1SingleEG20_TotemDiffractive_v1',
    'HLT_PAL1SingleJet16_v1',
    'HLT_PAL1SingleJet36_v1',
    'HLT_PAL1SingleJet52_TotemDiffractive_v1',
    'HLT_PAL1SingleMu20_TotemDiffractive_v1',
    'HLT_PAL1Tech53_MB_SingleTrack_v1',
    'HLT_PAL1Tech53_MB_v1',
    'HLT_PAL1Tech54_ZeroBias_v1',
    'HLT_PAL1Tech63_CASTORHaloMuon_v1',
    'HLT_PAL1Tech_HBHEHO_totalOR_v1',
    'HLT_PAL2DoubleMu3_v1',
    'HLT_PAMinBiasBHC_OR_v1',
    'HLT_PAMinBiasBHC_v1',
    'HLT_PAMinBiasHF_OR_v1',
    'HLT_PAMinBiasHF_v1',
    'HLT_PAMinBiasHfOrBHC_v1',
    'HLT_PAMu12_v1',
    'HLT_PAMu3PFJet20_v1',
    'HLT_PAMu3PFJet40_v1',
    'HLT_PAMu3_v1',
    'HLT_PAMu7PFJet20_v1',
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
    'HLT_PAMu7_v1',
    'HLT_PAPhoton10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton20_TightCaloIdVL_v1',
    'HLT_PAPhoton30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton30_TightCaloIdVL_v1',
    'HLT_PAPhoton40_NoCaloIdVL_v1',
    'HLT_PAPhoton40_TightCaloIdVL_v1',
    'HLT_PAPhoton60_NoCaloIdVL_v1',
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
    'HLT_PAPixelTracks_Multiplicity100_v2',
    'HLT_PAPixelTracks_Multiplicity130_v2',
    'HLT_PAPixelTracks_Multiplicity160_v2',
    'HLT_PAPixelTracks_Multiplicity190_v2',
    'HLT_PAPixelTracks_Multiplicity220_v2',
    'HLT_PARandom_v1',
    'HLT_PARomanPots_Tech52_v1',
    'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
    'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
    'HLT_PASingleForJet15_v1',
    'HLT_PASingleForJet25_v1',
    'HLT_PAT1minbias_Tech55_v1',
    'HLT_PATech35_HFSumET100_v3',
    'HLT_PATech35_v1',
    'HLT_PATripleJet100_20_20_v1',
    'HLT_PATripleJet20_20_20_v1',
    'HLT_PATripleJet40_20_20_v1',
    'HLT_PATripleJet60_20_20_v1',
    'HLT_PATripleJet80_20_20_v1',
    'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
    'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
    'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
    'HLT_PAZeroBiasPixel_DoubleTrack_v1',
    'HLT_PAZeroBiasPixel_SingleTrack_v1',
    'HLT_PAZeroBias_v1',
    'HLT_PPL1DoubleJetC36_v1',
    'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
    'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
    'HLT_PPPixelTracks_Multiplicity55_v1',
    'HLT_PPPixelTracks_Multiplicity70_v1',
    'HLT_PPPixelTracks_Multiplicity85_v1',
    'HLT_Physics_v5' ),
  PAHighPt = cms.vstring( 'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
    'HLT_PAForJet100Eta2_v1',
    'HLT_PAForJet100Eta3_v1',
    'HLT_PAForJet20Eta2_v1',
    'HLT_PAForJet20Eta3_v1',
    'HLT_PAForJet40Eta2_v1',
    'HLT_PAForJet40Eta3_v1',
    'HLT_PAForJet60Eta2_v1',
    'HLT_PAForJet60Eta3_v1',
    'HLT_PAForJet80Eta2_v1',
    'HLT_PAForJet80Eta3_v1',
    'HLT_PAFullTrack12_v2',
    'HLT_PAFullTrack20_v2',
    'HLT_PAFullTrack30_v2',
    'HLT_PAFullTrack50_v2',
    'HLT_PAHFSumET100_v3',
    'HLT_PAHFSumET140_v3',
    'HLT_PAHFSumET170_v3',
    'HLT_PAHFSumET210_v3',
    'HLT_PAJet100_NoJetID_v1',
    'HLT_PAJet120_NoJetID_v1',
    'HLT_PAJet20_NoJetID_v1',
    'HLT_PAJet40ETM30_v1',
    'HLT_PAJet40_NoJetID_v1',
    'HLT_PAJet60ETM30_v1',
    'HLT_PAJet60_NoJetID_v1',
    'HLT_PAJet80_NoJetID_v1',
    'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PAPhoton10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton20_TightCaloIdVL_v1',
    'HLT_PAPhoton30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton30_TightCaloIdVL_v1',
    'HLT_PAPhoton40_NoCaloIdVL_v1',
    'HLT_PAPhoton40_TightCaloIdVL_v1',
    'HLT_PAPhoton60_NoCaloIdVL_v1',
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
    'HLT_PAPixelTracks_Multiplicity100_v2',
    'HLT_PAPixelTracks_Multiplicity130_v2',
    'HLT_PAPixelTracks_Multiplicity160_v2',
    'HLT_PAPixelTracks_Multiplicity190_v2',
    'HLT_PAPixelTracks_Multiplicity220_v2',
    'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
    'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
    'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
    'HLT_PATech35_HFSumET100_v3',
    'HLT_PATech35_v1',
    'HLT_PATripleJet100_20_20_v1',
    'HLT_PATripleJet20_20_20_v1',
    'HLT_PATripleJet40_20_20_v1',
    'HLT_PATripleJet60_20_20_v1',
    'HLT_PATripleJet80_20_20_v1' ),
  PAMinBiasUPC = cms.vstring( 'HLT_PABptxMinusNotBptxPlus_v1',
    'HLT_PABptxPlusNotBptxMinus_v1',
    'HLT_PACastorEmNotHfCoincidencePm_v1',
    'HLT_PACastorEmNotHfSingleChannel_v1',
    'HLT_PACastorEmTotemLowMultiplicity_v1',
    'HLT_PADimuon0_NoVertexing_v1',
    'HLT_PADoubleJet20_ForwardBackward_v1',
    'HLT_PADoubleMu4_Acoplanarity03_v1',
    'HLT_PAExclDijet35_HFAND_v1',
    'HLT_PAHFOR_SingleTrack_v1',
    'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
    'HLT_PAL1DoubleEG3_FwdVeto_v1',
    'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
    'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
    'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
    'HLT_PAL1DoubleMu0_v1',
    'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
    'HLT_PAL1SingleEG20_TotemDiffractive_v1',
    'HLT_PAL1SingleJet16_v1',
    'HLT_PAL1SingleJet36_v1',
    'HLT_PAL1SingleJet52_TotemDiffractive_v1',
    'HLT_PAL1SingleMu20_TotemDiffractive_v1',
    'HLT_PAL1Tech53_MB_SingleTrack_v1',
    'HLT_PAL1Tech53_MB_v1',
    'HLT_PAL1Tech54_ZeroBias_v1',
    'HLT_PAL1Tech63_CASTORHaloMuon_v1',
    'HLT_PAL1Tech_HBHEHO_totalOR_v1',
    'HLT_PAMinBiasBHC_OR_v1',
    'HLT_PAMinBiasBHC_v1',
    'HLT_PAMinBiasHF_OR_v1',
    'HLT_PAMinBiasHF_v1',
    'HLT_PAMinBiasHfOrBHC_v1',
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
    'HLT_PARandom_v1',
    'HLT_PARomanPots_Tech52_v1',
    'HLT_PASingleForJet15_v1',
    'HLT_PASingleForJet25_v1',
    'HLT_PAT1minbias_Tech55_v1',
    'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
    'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
    'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
    'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
    'HLT_PAZeroBiasPixel_DoubleTrack_v1',
    'HLT_PAZeroBiasPixel_SingleTrack_v1',
    'HLT_PAZeroBias_v1' ),
  PAMuon = cms.vstring( 'HLT_PABTagMu_Jet20_Mu4_v1',
    'HLT_PAL1DoubleMu0_HighQ_v1',
    'HLT_PAL1DoubleMuOpen_v1',
    'HLT_PAL2DoubleMu3_v1',
    'HLT_PAMu12_v1',
    'HLT_PAMu3PFJet20_v1',
    'HLT_PAMu3PFJet40_v1',
    'HLT_PAMu3_v1',
    'HLT_PAMu7PFJet20_v1',
    'HLT_PAMu7_v1' ),
  PPFSQ = cms.vstring( 'HLT_PADimuon0_NoVertexing_v1',
    'HLT_PADoubleJet20_ForwardBackward_v1',
    'HLT_PADoubleMu4_Acoplanarity03_v1',
    'HLT_PAExclDijet35_HFAND_v1',
    'HLT_PAExclDijet35_HFOR_v1',
    'HLT_PAL1DoubleEG3_FwdVeto_v1',
    'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
    'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
    'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
    'HLT_PAL1DoubleMu0_v1',
    'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
    'HLT_PAL1SingleEG20_TotemDiffractive_v1',
    'HLT_PAL1SingleJet16_v1',
    'HLT_PAL1SingleJet36_v1',
    'HLT_PAL1SingleJet52_TotemDiffractive_v1',
    'HLT_PAL1SingleMu20_TotemDiffractive_v1',
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
    'HLT_PASingleForJet15_v1',
    'HLT_PASingleForJet25_v1',
    'HLT_PPL1DoubleJetC36_v1' ),
  PPJet = cms.vstring( 'HLT_PAForJet100Eta2_v1',
    'HLT_PAForJet100Eta3_v1',
    'HLT_PAForJet20Eta2_v1',
    'HLT_PAForJet20Eta3_v1',
    'HLT_PAForJet40Eta2_v1',
    'HLT_PAForJet40Eta3_v1',
    'HLT_PAForJet60Eta2_v1',
    'HLT_PAForJet60Eta3_v1',
    'HLT_PAForJet80Eta2_v1',
    'HLT_PAForJet80Eta3_v1',
    'HLT_PAFullTrack12_v2',
    'HLT_PAFullTrack20_v2',
    'HLT_PAFullTrack30_v2',
    'HLT_PAFullTrack50_v2',
    'HLT_PAHFSumET100_v3',
    'HLT_PAHFSumET140_v3',
    'HLT_PAHFSumET170_v3',
    'HLT_PAHFSumET210_v3',
    'HLT_PAJet100_NoJetID_v1',
    'HLT_PAJet120_NoJetID_v1',
    'HLT_PAJet20_NoJetID_v1',
    'HLT_PAJet40ETM30_v1',
    'HLT_PAJet40_NoJetID_v1',
    'HLT_PAJet60ETM30_v1',
    'HLT_PAJet60_NoJetID_v1',
    'HLT_PAJet80_NoJetID_v1',
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
    'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
    'HLT_PAPixelTracks_Multiplicity100_v2',
    'HLT_PAPixelTracks_Multiplicity130_v2',
    'HLT_PAPixelTracks_Multiplicity160_v2',
    'HLT_PAPixelTracks_Multiplicity190_v2',
    'HLT_PAPixelTracks_Multiplicity220_v2',
    'HLT_PATech35_HFSumET100_v3',
    'HLT_PATech35_v1',
    'HLT_PATripleJet100_20_20_v1',
    'HLT_PATripleJet20_20_20_v1',
    'HLT_PATripleJet40_20_20_v1',
    'HLT_PATripleJet60_20_20_v1',
    'HLT_PATripleJet80_20_20_v1',
    'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
    'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
    'HLT_PPPixelTracks_Multiplicity55_v1',
    'HLT_PPPixelTracks_Multiplicity70_v1',
    'HLT_PPPixelTracks_Multiplicity85_v1' ),
  PPMinBias = cms.vstring( 'HLT_PABptxMinusNotBptxPlus_v1',
    'HLT_PABptxPlusNotBptxMinus_v1',
    'HLT_PACastorEmNotHfCoincidencePm_v1',
    'HLT_PACastorEmNotHfSingleChannel_v1',
    'HLT_PACastorEmTotemLowMultiplicity_v1',
    'HLT_PAHFOR_SingleTrack_v1',
    'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
    'HLT_PAL1Tech53_MB_SingleTrack_v1',
    'HLT_PAL1Tech53_MB_v1',
    'HLT_PAL1Tech54_ZeroBias_v1',
    'HLT_PAL1Tech63_CASTORHaloMuon_v1',
    'HLT_PAL1Tech_HBHEHO_totalOR_v1',
    'HLT_PAMinBiasBHC_OR_v1',
    'HLT_PAMinBiasBHC_v1',
    'HLT_PAMinBiasHF_OR_v1',
    'HLT_PAMinBiasHF_v1',
    'HLT_PAMinBiasHfOrBHC_v1',
    'HLT_PARandom_v1',
    'HLT_PARomanPots_Tech52_v1',
    'HLT_PAT1minbias_Tech55_v1',
    'HLT_PAZeroBiasPixel_DoubleTrack_v1',
    'HLT_PAZeroBiasPixel_SingleTrack_v1',
    'HLT_PAZeroBias_v1' ),
  PPMuon = cms.vstring( 'HLT_Mu15_eta2p1_v5',
    'HLT_PABTagMu_Jet20_Mu4_v1',
    'HLT_PAL1DoubleMu0_HighQ_v1',
    'HLT_PAL1DoubleMuOpen_v1',
    'HLT_PAL2DoubleMu3_v1',
    'HLT_PAMu12_v1',
    'HLT_PAMu3PFJet20_v1',
    'HLT_PAMu3PFJet40_v1',
    'HLT_PAMu3_v1',
    'HLT_PAMu7PFJet20_v1',
    'HLT_PAMu7_v1' ),
  PPPhoton = cms.vstring( 'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
    'HLT_PAPhoton10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
    'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
    'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
    'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
    'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton20_TightCaloIdVL_v1',
    'HLT_PAPhoton30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
    'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
    'HLT_PAPhoton30_TightCaloIdVL_v1',
    'HLT_PAPhoton40_NoCaloIdVL_v1',
    'HLT_PAPhoton40_TightCaloIdVL_v1',
    'HLT_PAPhoton60_NoCaloIdVL_v1' ),
  RPCMonitor = cms.vstring( 'AlCa_RPCMuonNoHits_v9',
    'AlCa_RPCMuonNoTriggers_v9',
    'AlCa_RPCMuonNormalisation_v9' ),
  SingleElectron = cms.vstring( 'HLT_Ele22_CaloIdL_CaloIsoVL_v6' ),
  SingleMu = cms.vstring( 'HLT_Mu15_eta2p1_v5' ),
  TestEnablesEcalHcalDT = cms.vstring( 'HLT_DTCalibration_v2',
    'HLT_EcalCalibration_v3',
    'HLT_HcalCalibration_v3' ),
  TestEnablesTracker = cms.vstring( 'HLT_TrackerCalibration_v3' )
)

process.GlobalTag = cms.ESSource( "PoolDBESSource",
    globaltag = cms.string( "GR_H_V32::All" ),
    RefreshEachRun = cms.untracked.bool( True ),
    RefreshOpenIOVs = cms.untracked.bool( False ),
    toGet = cms.VPSet( 
    ),
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
    RefreshAlways = cms.untracked.bool( False ),
    connect = cms.string( "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG" ),
    ReconnectEachRun = cms.untracked.bool( True ),
    BlobStreamerName = cms.untracked.string( "TBufferBlobStreamingService" )
)
process.HepPDTESSource = cms.ESSource( "HepPDTESSource",
    pdtFileName = cms.FileInPath( "SimGeneral/HepPDTESSource/data/pythiaparticle.tbl" )
)
process.eegeom = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "EcalMappingRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.es_hardcode = cms.ESSource( "HcalHardcodeCalibrations",
    fromDDD = cms.untracked.bool( False ),
    toGet = cms.untracked.vstring( 'GainWidths' )
)
process.hltESSBTagRecord = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "JetTagComputerRecord" ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.magfield = cms.ESSource( "XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring( 'Geometry/CMSCommonData/data/normal/cmsextent.xml',
      'Geometry/CMSCommonData/data/cms.xml',
      'Geometry/CMSCommonData/data/cmsMagneticField.xml',
      'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
      'MagneticField/GeomBuilder/data/MagneticFieldParameters_07_2pi.xml' ),
    rootNodeName = cms.string( "cmsMagneticField:MAGF" )
)

process.hltIter4ESPMeasurementTrackerPA = cms.ESProducer( "MeasurementTrackerESProducer",
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
  stripClusterProducer = cms.string( "hltPAIter4SiStripClusters" ),
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
  ComponentName = cms.string( "hltIter4ESPMeasurementTrackerPA" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltPAIter4ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter4ESPTrajectoryBuilderITPA = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter4ESPTrajectoryBuilderITPA" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTrackerPA" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter4ESPPixelLayerPairsPA = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'TIB1+TIB2' ),
  ComponentName = cms.string( "hltIter4ESPPixelLayerPairsPA" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet(  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet(  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter4ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter4ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter4ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter4SiStripClustersReg" ),
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
  ComponentName = cms.string( "hltIter4ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter4ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter3ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter3ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter3SiStripClustersReg" ),
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
  ComponentName = cms.string( "hltIter3ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter3ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter3ESPLayerTripletsReg = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter3ESPLayerTripletsReg" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter2ESPPixelLayerPairsReg = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter2ESPPixelLayerPairsReg" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter2SiStripClustersReg" ),
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
  ComponentName = cms.string( "hltIter2ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter2ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter1ESPPixelLayerTripletsReg = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1ESPPixelLayerTripletsReg" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter1ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter1SiStripClustersReg" ),
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
  ComponentName = cms.string( "hltIter1ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter1ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltSiStripClustersReg" ),
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
  ComponentName = cms.string( "hltESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" )
)
process.AutoMagneticFieldESProducer = cms.ESProducer( "AutoMagneticFieldESProducer",
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
process.CSCGeometryESModule = cms.ESProducer( "CSCGeometryESModule",
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
process.CaloGeometryBuilder = cms.ESProducer( "CaloGeometryBuilder",
  SelectedCalos = cms.vstring( 'HCAL',
    'ZDC',
    'EcalBarrel',
    'EcalEndcap',
    'EcalPreshower',
    'TOWER' )
)
process.CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder" )
process.CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" )
)
process.CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  appendToDataLabel = cms.string( "" )
)
process.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "ClusterShapeHitFilter" )
)
process.DTGeometryESModule = cms.ESProducer( "DTGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
process.EcalBarrelGeometryFromDBEP = cms.ESProducer( "EcalBarrelGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalElectronicsMappingBuilder = cms.ESProducer( "EcalElectronicsMappingBuilder" )
process.EcalEndcapGeometryFromDBEP = cms.ESProducer( "EcalEndcapGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalLaserCorrectionService = cms.ESProducer( "EcalLaserCorrectionService" )
process.EcalPreshowerGeometryFromDBEP = cms.ESProducer( "EcalPreshowerGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalUnpackerWorkerESProducer = cms.ESProducer( "EcalUnpackerWorkerESProducer",
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
process.HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.HcalTopologyIdealEP = cms.ESProducer( "HcalTopologyIdealEP" )
process.MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.RPCGeometryESModule = cms.ESProducer( "RPCGeometryESModule",
  useDDD = cms.untracked.bool( False ),
  compatibiltyWith11 = cms.untracked.bool( True )
)
process.SiStripGainESProducer = cms.ESProducer( "SiStripGainESProducer",
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
process.SiStripQualityESProducer = cms.ESProducer( "SiStripQualityESProducer",
  appendToDataLabel = cms.string( "" ),
  PrintDebugOutput = cms.bool( False ),
  ThresholdForReducedGranularity = cms.double( 0.3 ),
  UseEmptyRunInfo = cms.bool( False ),
  ReduceGranularity = cms.bool( False ),
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
  )
)
process.SiStripRecHitMatcherESProducer = cms.ESProducer( "SiStripRecHitMatcherESProducer",
  ComponentName = cms.string( "StandardMatcher" ),
  NSigmaInside = cms.double( 3.0 )
)
process.SlaveField0 = cms.ESProducer( "UniformMagneticFieldESProducer",
  ZFieldInTesla = cms.double( 0.0 ),
  label = cms.untracked.string( "slave_0" )
)
process.SlaveField20 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "2_0T" ) ),
  label = cms.untracked.string( "slave_20" )
)
process.SlaveField30 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_0T" ) ),
  label = cms.untracked.string( "slave_30" )
)
process.SlaveField35 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_5T" ) ),
  label = cms.untracked.string( "slave_35" )
)
process.SlaveField38 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_8T" ) ),
  label = cms.untracked.string( "slave_38" )
)
process.SlaveField40 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "4_0T" ) ),
  label = cms.untracked.string( "slave_40" )
)
process.SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.StripCPEfromTrackAngleESProducer = cms.ESProducer( "StripCPEESProducer",
  TanDiffusionAngle = cms.double( 0.01 ),
  UncertaintyScaling = cms.double( 1.42 ),
  ThicknessRelativeUncertainty = cms.double( 0.02 ),
  MaybeNoiseThreshold = cms.double( 3.5 ),
  ComponentName = cms.string( "StripCPEfromTrackAngle" ),
  MinimumUncertainty = cms.double( 0.01 ),
  NoiseThreshold = cms.double( 2.3 )
)
process.TrackerDigiGeometryESModule = cms.ESProducer( "TrackerDigiGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
  fromDDD = cms.bool( False )
)
process.TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" )
)
process.VBF0 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
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
process.VBF20 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
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
process.VBF30 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
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
process.VBF35 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
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
process.VBF38 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
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
process.VBF40 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
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
process.ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" )
)
process.ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  includeBadChambers = cms.bool( False )
)
process.ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
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
process.hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
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
process.hcal_db_producer = cms.ESProducer( "HcalDbProducer" )
process.hltCombinedSecondaryVertex = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  trackPairV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.03 ) ),
  useTrackWeights = cms.bool( True ),
  useCategories = cms.bool( True ),
  pseudoMultiplicityMin = cms.uint32( 2 ),
  categoryVariableName = cms.string( "vertexCategory" ),
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
  correctVertexMass = cms.bool( True ),
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
process.hltESPAK5CaloL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1FastJetCorrectionESProducer',
    'hltESPL2RelativeCorrectionESProducer',
    'hltESPL3AbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAK5CaloL2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL2RelativeCorrectionESProducer',
    'hltESPL3AbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAK5PFL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1PFFastJetCorrectionESProducer',
    'hltESPL2PFRelativeCorrectionESProducer',
    'hltESPL3PFAbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAK5PFNoPUL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1PFNoPUFastJetCorrectionESProducer',
    'hltESPL2PFNoPURelativeCorrectionESProducer',
    'hltESPL3PFNoPUAbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
process.hltESPBwdAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPBwdAnalyticalPropagator" ),
  PropagationDirection = cms.string( "oppositeToMomentum" )
)
process.hltESPBwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPBwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPChi2EstimatorForRefit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2EstimatorForRefit" )
)
process.hltESPChi2MeasurementEstimator = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator" )
)
process.hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" )
)
process.hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" )
)
process.hltESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPCkf3HitTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPCkfTrajectoryBuilderForHI = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPCkfTrajectoryFilterForHI = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPCloseComponentsMerger5D = cms.ESProducer( "CloseComponentsMergerESProducer5D",
  ComponentName = cms.string( "hltESPCloseComponentsMerger5D" ),
  MaxComponents = cms.int32( 12 ),
  DistanceMeasure = cms.string( "hltESPKullbackLeiblerDistance5D" )
)
process.hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPESUnpackerWorker = cms.ESProducer( "ESUnpackerWorkerESProducer",
  RHAlgo = cms.PSet( 
    ESRecoAlgo = cms.int32( 0 ),
    Type = cms.string( "ESRecHitWorker" )
  ),
  DCCDataUnpacker = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) ),
  ComponentName = cms.string( "hltESPESUnpackerWorker" )
)
process.hltESPEcalRegionCablingESProducer = cms.ESProducer( "EcalRegionCablingESProducer",
  esMapping = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) )
)
process.hltESPEcalTrigTowerConstituentsMapBuilder = cms.ESProducer( "EcalTrigTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/EcalMapping/data/EndCap_TTMap.txt" )
)
process.hltESPElectronChi2 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 2000.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPElectronChi2" )
)
process.hltESPElectronMaterialEffects = cms.ESProducer( "GsfMaterialEffectsESProducer",
  BetheHeitlerParametrization = cms.string( "BetheHeitler_cdfmom_nC6_O5.par" ),
  EnergyLossUpdator = cms.string( "GsfBetheHeitlerUpdator" ),
  ComponentName = cms.string( "hltESPElectronMaterialEffects" ),
  MultipleScatteringUpdator = cms.string( "MultipleScatteringUpdator" ),
  Mass = cms.double( 5.11E-4 ),
  BetheHeitlerCorrection = cms.int32( 2 )
)
process.hltESPFastSteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPFwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPFwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPGlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )
process.hltESPGsfElectronFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPGsfTrajectoryFitter = cms.ESProducer( "GsfTrajectoryFitterESProducer",
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectoryFitter" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  GeometricalPropagator = cms.string( "hltESPAnalyticalPropagator" )
)
process.hltESPGsfTrajectorySmoother = cms.ESProducer( "GsfTrajectorySmootherESProducer",
  ErrorRescaling = cms.double( 100.0 ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectorySmoother" ),
  GeometricalPropagator = cms.string( "hltESPBwdAnalyticalPropagator" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" )
)
process.hltESPHIMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPHIPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPHIPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPHITTRHBuilderWithoutRefit = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "Fake" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "Fake" ),
  ComponentName = cms.string( "hltESPHITTRHBuilderWithoutRefit" )
)
process.hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" )
)
process.hltESPKullbackLeiblerDistance5D = cms.ESProducer( "DistanceBetweenComponentsESProducer5D",
  ComponentName = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  DistanceMeasure = cms.string( "KullbackLeibler" )
)
process.hltESPL1FastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6CaloJets','rho' ),
  algorithm = cms.string( "AK5CaloHLT" ),
  level = cms.string( "L1FastJet" )
)
process.hltESPL1PFFastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6PFJets','rho' ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L1FastJet" )
)
process.hltESPL1PFNoPUFastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6PFJets','rho' ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L1FastJet" )
)
process.hltESPL2PFNoPURelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L2Relative" )
)
process.hltESPL2PFRelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L2Relative" )
)
process.hltESPL2RelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5CaloHLT" ),
  level = cms.string( "L2Relative" )
)
process.hltESPL3AbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5CaloHLT" ),
  level = cms.string( "L3Absolute" )
)
process.hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPL3PFAbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L3Absolute" )
)
process.hltESPL3PFNoPUAbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L3Absolute" )
)
process.hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltESPMeasurementTrackerForHI = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltESPMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPMuTrackJpsiEffTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPMuTrackJpsiEffTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPMuTrackJpsiTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPMuTrackJpsiTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
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
process.hltESPMuonCkfTrajectoryBuilderSeedHit = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
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
process.hltESPMuonCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPMuonDetLayerGeometryESProducer = cms.ESProducer( "MuonDetLayerGeometryESProducer" )
process.hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
)
process.hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
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
process.hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False )
)
process.hltESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPPixelLayerTripletsHITHB = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPPixelLayerTripletsHITHE = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPPixelLayerTripletsReg = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
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
process.hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKSmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)
process.hltESPSiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
process.hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagator" )
)
process.hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAny" )
)
process.hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" )
)
process.hltESPSmartPropagatorOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorOpposite" )
)
process.hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 )
)
process.hltESPSoftLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer",
  ipSign = cms.string( "any" )
)
process.hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.hltESPStraightLinePropagator = cms.ESProducer( "StraightLinePropagatorESProducer",
  ComponentName = cms.string( "hltESPStraightLinePropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
process.hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" )
)
process.hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" )
)
process.hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" )
)
process.hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
)
process.hltESPTrackCounting3D1st = cms.ESProducer( "TrackCountingESProducer",
  deltaR = cms.double( -1.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  maximumDecayLength = cms.double( 5.0 ),
  nthTrack = cms.int32( 1 )
)
process.hltESPTrackCounting3D2nd = cms.ESProducer( "TrackCountingESProducer",
  deltaR = cms.double( -1.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  maximumDecayLength = cms.double( 5.0 ),
  nthTrack = cms.int32( 2 )
)
process.hltESPTrackerRecoGeometryESProducer = cms.ESProducer( "TrackerRecoGeometryESProducer",
  appendToDataLabel = cms.string( "" ),
  trackerGeometryLabel = cms.untracked.string( "" )
)
process.hltESPTrajectoryBuilderForElectrons = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPTrajectoryBuilderL3 = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( False )
)
process.hltESPTrajectoryCleanerBySharedSeeds = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedSeeds" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPTrajectoryFilterForElectrons = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPTrajectoryFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPbJetRegionalTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltESPbJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltHIAllESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltHIAllESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltHIAllESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltHIAllESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
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
process.hltHIAllESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter1ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter1ESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltIter1ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter1ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltIter3ESPLayerTripletsPA = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter3ESPLayerTripletsPA" ),
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
    skipClusters = cms.InputTag( "hltPAIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter3ESPTrajectoryBuilderITPA = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryBuilderITPA" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTrackerPA" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter3ESPMeasurementTrackerPA = cms.ESProducer( "MeasurementTrackerESProducer",
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
  stripClusterProducer = cms.string( "hltPAIter3SiStripClusters" ),
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
  ComponentName = cms.string( "hltIter3ESPMeasurementTrackerPA" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltPAIter3ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1ESPPixelLayerTripletsPA = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1ESPPixelLayerTripletsPA" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPPixelLayerPairsPA = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter2ESPPixelLayerPairsPA" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.006 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPTrajectoryBuilderITPA = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryBuilderITPA" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTrackerPA" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter2ESPMeasurementTrackerPA = cms.ESProducer( "MeasurementTrackerESProducer",
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
  stripClusterProducer = cms.string( "hltPAIter2SiStripClusters" ),
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
  ComponentName = cms.string( "hltIter2ESPMeasurementTrackerPA" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltPAIter2ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1ESPTrajectoryBuilderITPA = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryBuilderITPA" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTrackerPA" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter1ESPMeasurementTrackerPA = cms.ESProducer( "MeasurementTrackerESProducer",
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
  stripClusterProducer = cms.string( "hltPAIter1SiStripClusters" ),
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
  ComponentName = cms.string( "hltIter1ESPMeasurementTrackerPA" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltPAIter1ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter1Tau3MuESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltIter1Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter2ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter2ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltIter2ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter2ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltIter2Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter2Tau3MuESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltIter2Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter3ESPLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltIter3ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter3ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter3ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltIter3Tau3MuESPLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
process.hltIter3Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter3Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter4ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter4ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'TIB1+TIB2' ),
  ComponentName = cms.string( "hltIter4ESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet(  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet(  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter4ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hltIter4ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
process.hltIter4Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltIter4Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
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
process.hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  includeBadChambers = cms.bool( False )
)
process.navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" )
)
process.preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  includeBadChambers = cms.bool( False )
)
process.siPixelQualityESProducer = cms.ESProducer( "SiPixelQualityESProducer",
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiPixelQualityFromDbRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiPixelDetVOffRcd" ),
      tag = cms.string( "" )
    )
  )
)
process.siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer" )
process.siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
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
process.sistripconn = cms.ESProducer( "SiStripConnectivity" )

process.FastTimerService = cms.Service( "FastTimerService",
    dqmPath = cms.untracked.string( "HLT/TimerService" ),
    dqmModuleTimeRange = cms.untracked.double( 40.0 ),
    luminosityProduct = cms.untracked.InputTag( "hltScalersRawToDigi" ),
    enableTimingExclusive = cms.untracked.bool( False ),
    enableTimingModules = cms.untracked.bool( True ),
    enableDQMbyPathOverhead = cms.untracked.bool( False ),
    dqmTimeResolution = cms.untracked.double( 5.0 ),
    enableDQMbyModule = cms.untracked.bool( False ),
    dqmLuminosityResolution = cms.untracked.double( 1.0E31 ),
    skipFirstPath = cms.untracked.bool( False ),
    enableTimingPaths = cms.untracked.bool( True ),
    enableDQMbyLumiSection = cms.untracked.bool( True ),
    dqmPathTimeResolution = cms.untracked.double( 0.5 ),
    dqmPathTimeRange = cms.untracked.double( 100.0 ),
    dqmTimeRange = cms.untracked.double( 1000.0 ),
    dqmLumiSectionsRange = cms.untracked.uint32( 2500 ),
    enableDQMSummary = cms.untracked.bool( True ),
    enableTimingSummary = cms.untracked.bool( False ),
    enableDQMbyPathTotal = cms.untracked.bool( False ),
    useRealTimeClock = cms.untracked.bool( True ),
    enableDQMbyPathExclusive = cms.untracked.bool( False ),
    enableDQMbyLuminosity = cms.untracked.bool( True ),
    enableDQM = cms.untracked.bool( True ),
    supportedProcesses = cms.untracked.vuint32( 8, 12, 16, 24, 32 ),
    dqmModuleTimeResolution = cms.untracked.double( 0.2 ),
    dqmLuminosityRange = cms.untracked.double( 1.0E34 ),
    enableDQMbyPathActive = cms.untracked.bool( False ),
    enableDQMbyPathDetails = cms.untracked.bool( False ),
    enableDQMbyProcesses = cms.untracked.bool( True ),
    enableDQMbyPathCounters = cms.untracked.bool( False )
)
process.DQM = cms.Service( "DQM",
    publishFrequency = cms.untracked.double( 5.0 ),
    debug = cms.untracked.bool( False ),
    collectorPort = cms.untracked.int32( 0 ),
    collectorHost = cms.untracked.string( "" )
)
process.DQMStore = cms.Service( "DQMStore",
)
process.DTDataIntegrityTask = cms.Service( "DTDataIntegrityTask",
    processingMode = cms.untracked.string( "HLT" ),
    fedIntegrityFolder = cms.untracked.string( "DT/FEDIntegrity_EvF" ),
    getSCInfo = cms.untracked.bool( True )
)
process.MessageLogger = cms.Service( "MessageLogger",
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
process.MicroStateService = cms.Service( "MicroStateService",
)
process.ModuleWebRegistry = cms.Service( "ModuleWebRegistry",
)
process.PrescaleService = cms.Service( "PrescaleService",
    forceDefault = cms.bool( False ),
    prescaleTable = cms.VPSet( 
      cms.PSet(  pathName = cms.string( "HLT_Activity_Ecal_SC7_v13" ),
        prescales = cms.vuint32( 280, 280, 280, 280, 280, 280, 280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_Mu15_eta2p1_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_Ele22_CaloIdL_CaloIsoVL_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_BeamGas_HF_Beam1_v5" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_BeamGas_HF_Beam2_v5" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_BeamHalo_v13" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHcalUTCA_v1" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHcalPhiSym_v1" ),
        prescales = cms.vuint32( 15, 15, 15, 15, 15, 15, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHcalNZS_v1" ),
        prescales = cms.vuint32( 15, 15, 15, 15, 15, 15, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_GlobalRunHPDNoise_v8" ),
        prescales = cms.vuint32( 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 0, 0, 0, 40, 40 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_Physics_v5" ),
        prescales = cms.vuint32( 8000, 8000, 8000, 8000, 8000, 8000, 8000, 3000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 0, 0, 0, 80, 80 )
      ),
      cms.PSet(  pathName = cms.string( "DST_Physics_v5" ),
        prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 10, 10 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_DTCalibration_v2" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_EcalCalibration_v3" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HcalCalibration_v3" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_TrackerCalibration_v3" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_L1SingleMuOpen_AntiBPTX_v7" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 10, 10, 10, 10, 10, 8, 8, 6, 10, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_L1TrackerCosmics_v7" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_PAEcalPi0EBonly_v1" ),
        prescales = cms.vuint32( 3, 3, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_PAEcalPi0EEonly_v1" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_PAEcalEtaEBonly_v1" ),
        prescales = cms.vuint32( 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_PAEcalEtaEEonly_v1" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_EcalPhiSym_v13" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 1, 1, 1, 1, 2668, 2668, 2668, 2668, 2668, 2668, 2668, 2668, 2668, 60, 30, 30, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_RPCMuonNoTriggers_v9" ),
        prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_RPCMuonNoHits_v9" ),
        prescales = cms.vuint32( 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_RPCMuonNormalisation_v9" ),
        prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_LumiPixels_v8" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_LumiPixels_ZeroBias_v4" ),
        prescales = cms.vuint32( 18, 18, 18, 18, 18, 18, 18, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1113, 50, 50, 50, 50, 101, 101, 101, 101, 101, 101, 101, 101, 101, 1113, 550, 550, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "AlCa_LumiPixels_Random_v1" ),
        prescales = cms.vuint32( 30, 30, 30, 30, 30, 30, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1SingleJet16_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 800, 800, 800, 800, 680, 800, 960, 800, 560, 360, 240, 160, 45000, 10, 1, 10, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1SingleJet36_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 400, 400, 320, 200, 140, 80, 48, 40, 28, 18, 12, 8, 2000, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PASingleForJet15_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 10, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PASingleForJet25_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 220, 220, 176, 110, 77, 44, 26, 22, 15, 10, 7, 5, 100, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet20_NoJetID_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4549, 4549, 4549, 4549, 3515, 3381, 4057, 4959, 4388, 3347, 2231, 1487, 503, 10, 1, 10, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet40_NoJetID_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 127, 127, 127, 127, 108, 103, 121, 103, 72, 47, 31, 21, 29, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet60_NoJetID_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 173, 173, 138, 86, 60, 26, 15, 13, 9, 5, 3, 2, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet80_NoJetID_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 41, 41, 33, 20, 14, 6, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet100_NoJetID_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 13, 13, 10, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet120_NoJetID_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet20Eta2_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2588, 2588, 2588, 2588, 800, 763, 916, 1076, 979, 859, 741, 599, 151, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet40Eta2_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 547, 547, 437, 273, 70, 43, 26, 25, 24, 23, 22, 18, 7, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet60Eta2_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 61, 61, 49, 30, 12, 6, 4, 3, 3, 3, 2, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet80Eta2_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 11, 11, 9, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet100Eta2_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet20Eta3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2062, 2062, 2062, 2062, 316, 300, 360, 422, 389, 341, 294, 239, 37, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet40Eta3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 307, 307, 246, 153, 31, 13, 7, 8, 8, 5, 14, 5, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet60Eta3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 29, 29, 23, 14, 10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet80Eta3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAForJet100Eta3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PATripleJet20_20_20_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 349, 349, 349, 349, 149, 132, 158, 188, 167, 148, 132, 105, 5, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PATripleJet40_20_20_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 449, 449, 359, 224, 150, 59, 35, 36, 31, 24, 17, 13, 5, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PATripleJet60_20_20_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 89, 89, 71, 44, 48, 22, 12, 11, 9, 5, 4, 3, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PATripleJet80_20_20_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 23, 23, 18, 11, 16, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PATripleJet100_20_20_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet40ETM30_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAJet60ETM30_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleMu0_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 30, 30, 24, 15, 11, 10, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PADimuon0_NoVertexing_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleMu0_HighQ_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleMuOpen_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL2DoubleMu3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMu3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 30, 30, 24, 15, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMu7_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 9, 9, 7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMu12_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PABTagMu_Jet20_Mu4_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMu3PFJet20_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 11, 11, 9, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMu3PFJet40_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMu7PFJet20_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton10_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 200, 200, 200, 140, 240, 90, 112, 100, 70, 45, 30, 20, 5, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton15_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 80, 80, 80, 56, 19, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton20_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 20, 20, 20, 14, 5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton30_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton40_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton60_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton10_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 80, 80, 80, 56, 70, 40, 50, 30, 24, 16, 11, 7, 4, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton15_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 10, 7, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton20_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton30_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton40_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton10_TightCaloIdVL_Iso50_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 80, 80, 80, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton15_TightCaloIdVL_Iso50_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 10, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton20_TightCaloIdVL_Iso50_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton30_TightCaloIdVL_Iso50_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton10_Photon10_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 100, 100, 80, 50, 35, 10, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton15_Photon10_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 20, 20, 16, 10, 7, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton20_Photon15_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 4, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton20_Photon20_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton30_Photon30_NoCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton10_Photon10_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 6, 6, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton15_Photon10_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPhoton20_Photon15_TightCaloIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PASingleEle6_CaloIdT_TrkIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 100, 100, 80, 50, 35, 20, 12, 10, 7, 5, 4, 2, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1000, 1000, 800, 500, 350, 200, 120, 100, 70, 45, 30, 20, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 500, 500, 400, 250, 175, 100, 60, 50, 35, 23, 15, 10, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTracks_Multiplicity100_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 100, 100, 100, 100, 150, 150, 180, 150, 210, 135, 90, 60, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTracks_Multiplicity130_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 20, 20, 20, 20, 30, 30, 36, 30, 42, 27, 18, 12, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTracks_Multiplicity160_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 5, 6, 8, 10, 8, 6, 4, 3, 2, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTracks_Multiplicity190_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTracks_Multiplicity220_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTrackMultiplicity100_FullTrack12_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 4, 4, 4, 4, 6, 4, 3, 2, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTrackMultiplicity130_FullTrack12_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTrackMultiplicity160_FullTrack12_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAFullTrack12_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 20, 20, 20, 20, 34, 40, 48, 40, 28, 18, 12, 8, 100, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAFullTrack20_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 5, 4, 5, 6, 5, 4, 2, 1, 1, 10, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAFullTrack30_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAFullTrack50_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PPPixelTracks_Multiplicity55_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PPPixelTracks_Multiplicity70_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PPPixelTracks_Multiplicity85_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PPPixelTrackMultiplicity55_FullTrack12_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PPPixelTrackMultiplicity70_FullTrack12_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PPL1DoubleJetC36_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PATech35_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 315, 210, 126, 84, 59, 38, 25, 17, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PATech35_HFSumET100_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 315, 210, 126, 84, 59, 38, 25, 17, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHFSumET100_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 200, 200, 240, 200, 280, 180, 120, 80, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHFSumET140_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 20, 20, 24, 20, 28, 18, 12, 8, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHFSumET170_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHFSumET210_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PARomanPots_Tech52_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1Tech53_MB_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1Tech53_MB_SingleTrack_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1Tech54_ZeroBias_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAT1minbias_Tech55_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1Tech_HBHEHO_totalOR_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 10, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1Tech63_CASTORHaloMuon_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PACastorEmTotemLowMultiplicity_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PACastorEmNotHfCoincidencePm_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PACastorEmNotHfSingleChannel_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1CastorTotalTotemLowMultiplicity_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMinBiasHF_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMinBiasHF_OR_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 10, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMinBiasBHC_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMinBiasBHC_OR_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMinBiasHfOrBHC_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PABptxPlusNotBptxMinus_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PABptxMinusNotBptxPlus_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAZeroBias_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 191, 20, 20, 20, 20, 80, 80, 80, 80, 80, 80, 80, 80, 80, 191, 97, 97, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAZeroBiasPixel_SingleTrack_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 8, 5, 16, 8, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAHFOR_SingleTrack_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAZeroBiasPixel_DoubleTrack_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 100, 100, 80, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PADoubleMu4_Acoplanarity03_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAExclDijet35_HFOR_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAExclDijet35_HFAND_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleEG3_FwdVeto_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1SingleJet52_TotemDiffractive_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1SingleMu20_TotemDiffractive_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1SingleEG20_TotemDiffractive_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleJet20_TotemDiffractive_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleJetC36_TotemDiffractive_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleMu5_TotemDiffractive_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAL1DoubleEG5_TotemDiffractive_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PADoubleJet20_ForwardBackward_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAUpcSingleEG5Pixel_TrackVeto_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 6, 4, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAUpcSingleEG5Full_TrackVeto7_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 30, 30, 24, 15, 10, 6, 3, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PAUpcSingleMuOpenTkMu_Onia_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_PARandom_v1" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 60, 60, 60, 60, 60, 60, 60, 60, 60, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "DQM_FEDIntegrity_v11" ),
        prescales = cms.vuint32( 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_LogMonitor_v4" ),
        prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "AOutput" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 )
      ),
      cms.PSet(  pathName = cms.string( "ALCALUMIPIXELSOutput" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "DQMOutput" ),
        prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2, 20, 20, 20, 20, 33, 33, 33, 33, 33, 33, 33, 33, 33, 2, 2, 2, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "ExpressOutput" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1, 1, 1, 32, 1 )
      )
    ),
    lvl1DefaultLabel = cms.string( "3e33" ),
    lvl1Labels = cms.vstring( '9e33nopark',
      '8e33nopark',
      '8e33',
      '7e33',
      '6e33',
      '4e33',
      '2e33',
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
      'PAPilot8Bunches',
      'PAEM',
      'PA2MHz',
      'PA1600kHz',
      'PA1100kHz',
      'PA750kHz',
      'PA500kHz',
      'PA300kHz',
      'PA200kHz',
      'PA140kHz',
      'PA90kHz',
      'PA60kHz',
      'PA40kHz',
      'PP4MHz',
      'PAPilot8BunchesEM',
      'PAPilot4Bunches',
      'PAPilot4BunchesEM',
      'CirculatingBeam',
      'CirculatingBeam+HighRandom' )
)
process.UpdaterService = cms.Service( "UpdaterService",
)

process.hltGetConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
    toGet = cms.VPSet( 
    ),
    verbose = cms.untracked.bool( False )
)
process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
process.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
process.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
process.hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtFedId = cms.untracked.int32( 813 ),
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    UnpackBxInEvent = cms.int32( 5 ),
    ActiveBoardsMask = cms.uint32( 0xffff )
)
process.hltGctDigis = cms.EDProducer( "GctRawToDigi",
    unpackSharedRegions = cms.bool( False ),
    numberOfGctSamplesToUnpack = cms.uint32( 1 ),
    verbose = cms.untracked.bool( False ),
    numberOfRctSamplesToUnpack = cms.uint32( 1 ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    unpackerVersion = cms.uint32( 0 ),
    gctFedId = cms.untracked.int32( 745 ),
    hltMode = cms.bool( True )
)
process.hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
    TechnicalTriggersUnprescaled = cms.bool( True ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    AlgorithmTriggersUnmasked = cms.bool( False ),
    EmulateBxInEvent = cms.int32( 1 ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    ProduceL1GtDaqRecord = cms.bool( False ),
    ReadTechnicalTriggerRecords = cms.bool( True ),
    RecordLength = cms.vint32( 3, 0 ),
    TechnicalTriggersUnmasked = cms.bool( False ),
    ProduceL1GtEvmRecord = cms.bool( False ),
    GmtInputTag = cms.InputTag( "hltGtDigis" ),
    TechnicalTriggersVetoUnmasked = cms.bool( True ),
    AlternativeNrBxBoardEvm = cms.uint32( 0 ),
    TechnicalTriggersInputTags = cms.VInputTag( 'simBscDigis' ),
    CastorInputTag = cms.InputTag( "castorL1Digis" ),
    GctInputTag = cms.InputTag( "hltGctDigis" ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    WritePsbL1GtDaqRecord = cms.bool( False ),
    BstLengthBytes = cms.int32( -1 )
)
process.hltL1extraParticles = cms.EDProducer( "L1ExtraParticlesProd",
    tauJetSource = cms.InputTag( 'hltGctDigis','tauJets' ),
    etHadSource = cms.InputTag( "hltGctDigis" ),
    etTotalSource = cms.InputTag( "hltGctDigis" ),
    centralBxOnly = cms.bool( True ),
    centralJetSource = cms.InputTag( 'hltGctDigis','cenJets' ),
    etMissSource = cms.InputTag( "hltGctDigis" ),
    hfRingEtSumsSource = cms.InputTag( "hltGctDigis" ),
    produceMuonParticles = cms.bool( True ),
    forwardJetSource = cms.InputTag( 'hltGctDigis','forJets' ),
    ignoreHtMiss = cms.bool( False ),
    htMissSource = cms.InputTag( "hltGctDigis" ),
    produceCaloParticles = cms.bool( True ),
    muonSource = cms.InputTag( "hltGtDigis" ),
    isolatedEmSource = cms.InputTag( 'hltGctDigis','isoEm' ),
    nonIsolatedEmSource = cms.InputTag( 'hltGctDigis','nonIsoEm' ),
    hfRingBitCountsSource = cms.InputTag( "hltGctDigis" )
)
process.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
process.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    maxZ = cms.double( 40.0 ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    gtEvmLabel = cms.InputTag( "" ),
    changeToCMSCoordinates = cms.bool( False ),
    setSigmaZ = cms.double( 0.0 ),
    maxRadius = cms.double( 2.0 )
)
process.hltL1sL1ZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreActivityEcalSC7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalRawToRecHitFacility = cms.EDProducer( "EcalRawToRecHitFacility",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    workerName = cms.string( "" )
)
process.hltESRawToRecHitFacility = cms.EDProducer( "EcalRawToRecHitFacility",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    workerName = cms.string( "hltESPESUnpackerWorker" )
)
process.hltEcalRegionalRestFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet( 
    ),
    sourceTag_es = cms.InputTag( "NotNeededoESfalse" ),
    doES = cms.bool( False ),
    type = cms.string( "all" ),
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    EmJobPSet = cms.VPSet( 
    ),
    CandJobPSet = cms.VPSet( 
    ),
    MuonJobPSet = cms.PSet(  ),
    esInstance = cms.untracked.string( "es" ),
    MuJobPSet = cms.PSet(  )
)
process.hltEcalRegionalESRestFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet( 
    ),
    sourceTag_es = cms.InputTag( "hltESRawToRecHitFacility" ),
    doES = cms.bool( True ),
    type = cms.string( "all" ),
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    EmJobPSet = cms.VPSet( 
    ),
    CandJobPSet = cms.VPSet( 
    ),
    MuonJobPSet = cms.PSet(  ),
    esInstance = cms.untracked.string( "es" ),
    MuJobPSet = cms.PSet(  )
)
process.hltEcalRecHitAll = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( True ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    sourceTag = cms.InputTag( "hltEcalRegionalRestFEDs" ),
    cleaningConfig = cms.PSet( 
      e6e2thresh = cms.double( 0.04 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      tightenCrack_e1_double = cms.double( 2.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      ignoreOutOfTimeThresh = cms.double( 1.0E9 ),
      cThreshold_endcap = cms.double( 15.0 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      cThreshold_double = cms.double( 10.0 )
    ),
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" )
)
process.hltESRecHitAll = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( False ),
    rechitCollection = cms.string( "EcalRecHitsES" ),
    EErechitCollection = cms.string( "" ),
    EBrechitCollection = cms.string( "" ),
    sourceTag = cms.InputTag( 'hltEcalRegionalESRestFEDs','es' ),
    cleaningConfig = cms.PSet(  ),
    lazyGetterTag = cms.InputTag( "hltESRawToRecHitFacility" )
)
process.hltHybridSuperClustersActivity = cms.EDProducer( "HybridClusterProducer",
    eThreshA = cms.double( 0.003 ),
    basicclusterCollection = cms.string( "hybridBarrelBasicClusters" ),
    clustershapecollection = cms.string( "" ),
    ethresh = cms.double( 0.1 ),
    ewing = cms.double( 0.0 ),
    RecHitSeverityToBeExcluded = cms.vstring( 'kWeird' ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    ),
    HybridBarrelSeedThr = cms.double( 1.0 ),
    dynamicPhiRoad = cms.bool( False ),
    shapeAssociation = cms.string( "hybridShapeAssoc" ),
    RecHitFlagToBeExcluded = cms.vstring(  ),
    useEtForXi = cms.bool( True ),
    step = cms.int32( 17 ),
    eThreshB = cms.double( 0.1 ),
    xi = cms.double( 0.0 ),
    eseed = cms.double( 0.35 ),
    ecalhitproducer = cms.string( "hltEcalRecHitAll" ),
    dynamicEThresh = cms.bool( False ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    excludeFlagged = cms.bool( True ),
    superclusterCollection = cms.string( "" )
)
process.hltCorrectedHybridSuperClustersActivity = cms.EDProducer( "EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    superClusterAlgo = cms.string( "Hybrid" ),
    etThresh = cms.double( 5.0 ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersActivity" ),
    applyEnergyCorrection = cms.bool( True ),
    isl_fCorrPset = cms.PSet(  ),
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    fix_fCorrPset = cms.PSet(  ),
    modeEE = cms.int32( 0 ),
    modeEB = cms.int32( 0 ),
    dyn_fCorrPset = cms.PSet(  ),
    energyCorrectorName = cms.string( "EcalClusterEnergyCorrectionObjectSpecific" ),
    applyCrackCorrection = cms.bool( False ),
    hyb_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 1.1 ),
      fEtEtaVec = cms.vdouble( 0.0, 1.00121, -0.63672, 0.0, 0.0, 0.0, 0.5655, 6.457, 0.5081, 8.0, 1.023, -0.00181 ),
      brLinearHighThr = cms.double( 8.0 ),
      fBremVec = cms.vdouble( -0.04382, 0.1169, 0.9267, -9.413E-4, 1.419 )
    )
)
process.hltMulti5x5BasicClustersActivity = cms.EDProducer( "Multi5x5ClusterProducer",
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "multi5x5BarrelBasicClusters" ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    doEndcap = cms.bool( True ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    ),
    barrelShapeAssociation = cms.string( "multi5x5BarrelShapeAssoc" ),
    endcapShapeAssociation = cms.string( "multi5x5EndcapShapeAssoc" ),
    endcapHitProducer = cms.string( "hltEcalRecHitAll" ),
    clustershapecollectionEB = cms.string( "multi5x5BarrelShape" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    barrelHitProducer = cms.string( "hltEcalRecHitAll" ),
    RecHitFlagToBeExcluded = cms.vstring(  ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    clustershapecollectionEE = cms.string( "multi5x5EndcapShape" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    doBarrel = cms.bool( False )
)
process.hltMulti5x5SuperClustersActivity = cms.EDProducer( "Multi5x5SuperClusterProducer",
    barrelSuperclusterCollection = cms.string( "multi5x5BarrelSuperClusters" ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    barrelClusterCollection = cms.string( "multi5x5BarrelBasicClusters" ),
    dynamicPhiRoad = cms.bool( False ),
    endcapClusterProducer = cms.string( "hltMulti5x5BasicClustersActivity" ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    barrelClusterProducer = cms.string( "hltMulti5x5BasicClustersActivity" ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    endcapSuperclusterCollection = cms.string( "multi5x5EndcapSuperClusters" ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    bremRecoveryPset = cms.PSet( 
      barrel = cms.PSet( 
        cryVec = cms.vint32( 16, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3 ),
        cryMin = cms.int32( 2 ),
        etVec = cms.vdouble( 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 45.0, 55.0, 135.0, 195.0, 225.0 )
      ),
      endcap = cms.PSet( 
        a = cms.double( 47.85 ),
        c = cms.double( 0.1201 ),
        b = cms.double( 108.8 )
      )
    ),
    doEndcaps = cms.bool( True ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    doBarrel = cms.bool( False )
)
process.hltMulti5x5SuperClustersWithPreshowerActivity = cms.EDProducer( "PreshowerClusterProducer",
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    etThresh = cms.double( 0.0 ),
    preshRecHitProducer = cms.InputTag( 'hltESRecHitAll','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( 'hltMulti5x5SuperClustersActivity','multi5x5EndcapSuperClusters' ),
    preshNclust = cms.int32( 4 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 )
)
process.hltCorrectedMulti5x5SuperClustersWithPreshowerActivity = cms.EDProducer( "EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    superClusterAlgo = cms.string( "Multi5x5" ),
    etThresh = cms.double( 5.0 ),
    rawSuperClusterProducer = cms.InputTag( "hltMulti5x5SuperClustersWithPreshowerActivity" ),
    applyEnergyCorrection = cms.bool( True ),
    isl_fCorrPset = cms.PSet(  ),
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    fix_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.9 ),
      fEtEtaVec = cms.vdouble( 1.0, -0.4386, -32.38, 0.6372, 15.67, -0.0928, -2.462, 1.138, 20.93 ),
      brLinearHighThr = cms.double( 6.0 ),
      fBremVec = cms.vdouble( -0.05228, 0.08738, 0.9508, 0.002677, 1.221 )
    ),
    modeEE = cms.int32( 0 ),
    modeEB = cms.int32( 0 ),
    dyn_fCorrPset = cms.PSet(  ),
    energyCorrectorName = cms.string( "EcalClusterEnergyCorrectionObjectSpecific" ),
    applyCrackCorrection = cms.bool( False ),
    hyb_fCorrPset = cms.PSet(  )
)
process.hltRecoEcalSuperClusterActivityCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedMulti5x5SuperClustersWithPreshowerActivity" ),
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersActivity" ),
    recoEcalCandidateCollection = cms.string( "" )
)
process.hltEcalActivitySuperClusterWrapper = cms.EDFilter( "HLTEgammaTriggerFilterObjectWrapper",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    candIsolatedTag = cms.InputTag( "hltRecoEcalSuperClusterActivityCandidate" ),
    candNonIsolatedTag = cms.InputTag( "" )
)
process.hltEgammaSelectEcalSuperClustersActivityFilterSC7 = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltRecoEcalSuperClusterActivityCandidate" ),
    inputTag = cms.InputTag( "hltEcalActivitySuperClusterWrapper" ),
    etcutEB = cms.double( 7.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 7.0 )
)
process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltL1sL1SingleMu7 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreMu15eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1fL1sMu7L1FilteredEta2p1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu7" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
    useStandardFEDid = cms.bool( True ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    dataType = cms.string( "DDU" ),
    fedbyType = cms.bool( False ),
    readOutParameters = cms.PSet( 
      debug = cms.untracked.bool( False ),
      rosParameters = cms.PSet( 
        writeSC = cms.untracked.bool( True ),
        readingDDU = cms.untracked.bool( True ),
        performDataIntegrityMonitor = cms.untracked.bool( False ),
        readDDUIDfromDDU = cms.untracked.bool( True ),
        debug = cms.untracked.bool( False ),
        localDAQ = cms.untracked.bool( False )
      ),
      localDAQ = cms.untracked.bool( False ),
      performDataIntegrityMonitor = cms.untracked.bool( False )
    ),
    dqmOnly = cms.bool( False )
)
process.hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    debug = cms.untracked.bool( False ),
    recAlgoConfig = cms.PSet( 
      tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
      minTime = cms.double( -3.0 ),
      stepTwoFromDigi = cms.bool( False ),
      doVdriftCorr = cms.bool( False ),
      debug = cms.untracked.bool( False ),
      maxTime = cms.double( 420.0 ),
      tTrigModeConfig = cms.PSet( 
        vPropWire = cms.double( 24.4 ),
        doTOFCorrection = cms.bool( True ),
        tofCorrType = cms.int32( 0 ),
        wirePropCorrType = cms.int32( 0 ),
        tTrigLabel = cms.string( "" ),
        doWirePropCorrection = cms.bool( True ),
        doT0Correction = cms.bool( True ),
        debug = cms.untracked.bool( False )
      )
    ),
    dtDigiLabel = cms.InputTag( "hltMuonDTDigis" ),
    recAlgo = cms.string( "DTLinearDriftFromDBAlgo" )
)
process.hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
    debug = cms.untracked.bool( False ),
    Reco4DAlgoName = cms.string( "DTCombinatorialPatternReco4D" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
    Reco4DAlgoConfig = cms.PSet( 
      segmCleanerMode = cms.int32( 2 ),
      Reco2DAlgoName = cms.string( "DTCombinatorialPatternReco" ),
      recAlgoConfig = cms.PSet( 
        tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
        minTime = cms.double( -3.0 ),
        stepTwoFromDigi = cms.bool( False ),
        doVdriftCorr = cms.bool( False ),
        debug = cms.untracked.bool( False ),
        maxTime = cms.double( 420.0 ),
        tTrigModeConfig = cms.PSet( 
          vPropWire = cms.double( 24.4 ),
          doTOFCorrection = cms.bool( True ),
          tofCorrType = cms.int32( 0 ),
          wirePropCorrType = cms.int32( 0 ),
          tTrigLabel = cms.string( "" ),
          doWirePropCorrection = cms.bool( True ),
          doT0Correction = cms.bool( True ),
          debug = cms.untracked.bool( False )
        )
      ),
      nSharedHitsMax = cms.int32( 2 ),
      hit_afterT0_resolution = cms.double( 0.03 ),
      Reco2DAlgoConfig = cms.PSet( 
        segmCleanerMode = cms.int32( 2 ),
        recAlgoConfig = cms.PSet( 
          tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
          minTime = cms.double( -3.0 ),
          stepTwoFromDigi = cms.bool( False ),
          doVdriftCorr = cms.bool( False ),
          debug = cms.untracked.bool( False ),
          maxTime = cms.double( 420.0 ),
          tTrigModeConfig = cms.PSet( 
            vPropWire = cms.double( 24.4 ),
            doTOFCorrection = cms.bool( True ),
            tofCorrType = cms.int32( 0 ),
            wirePropCorrType = cms.int32( 0 ),
            tTrigLabel = cms.string( "" ),
            doWirePropCorrection = cms.bool( True ),
            doT0Correction = cms.bool( True ),
            debug = cms.untracked.bool( False )
          )
        ),
        nSharedHitsMax = cms.int32( 2 ),
        AlphaMaxPhi = cms.double( 1.0 ),
        hit_afterT0_resolution = cms.double( 0.03 ),
        MaxAllowedHits = cms.uint32( 50 ),
        performT0_vdriftSegCorrection = cms.bool( False ),
        AlphaMaxTheta = cms.double( 0.9 ),
        debug = cms.untracked.bool( False ),
        recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
        nUnSharedHitsMin = cms.int32( 2 ),
        performT0SegCorrection = cms.bool( False ),
        perform_delta_rejecting = cms.bool( False )
      ),
      performT0_vdriftSegCorrection = cms.bool( False ),
      debug = cms.untracked.bool( False ),
      recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
      nUnSharedHitsMin = cms.int32( 2 ),
      AllDTRecHits = cms.bool( True ),
      performT0SegCorrection = cms.bool( False ),
      perform_delta_rejecting = cms.bool( False )
    )
)
process.hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool( False ),
    UseSelectiveUnpacking = cms.bool( True ),
    UseExaminer = cms.bool( True ),
    ErrorMask = cms.uint32( 0x0 ),
    InputObjects = cms.InputTag( "rawDataCollector" ),
    UseFormatStatus = cms.bool( True ),
    ExaminerMask = cms.uint32( 0x1febf3f6 ),
    UnpackStatusDigis = cms.bool( False ),
    VisualFEDInspect = cms.untracked.bool( False ),
    FormatedEventDump = cms.untracked.bool( False ),
    Debug = cms.untracked.bool( False ),
    VisualFEDShort = cms.untracked.bool( False )
)
process.hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    XTasymmetry_ME1b = cms.double( 0.0 ),
    XTasymmetry_ME1a = cms.double( 0.0 ),
    ConstSyst_ME1a = cms.double( 0.022 ),
    ConstSyst_ME1b = cms.double( 0.007 ),
    XTasymmetry_ME41 = cms.double( 0.0 ),
    CSCStripxtalksOffset = cms.double( 0.03 ),
    CSCUseCalibrations = cms.bool( True ),
    CSCUseTimingCorrections = cms.bool( True ),
    CSCNoOfTimeBinsForDynamicPedestal = cms.int32( 2 ),
    XTasymmetry_ME22 = cms.double( 0.0 ),
    UseFivePoleFit = cms.bool( True ),
    XTasymmetry_ME21 = cms.double( 0.0 ),
    ConstSyst_ME21 = cms.double( 0.0 ),
    CSCDebug = cms.untracked.bool( False ),
    ConstSyst_ME22 = cms.double( 0.0 ),
    CSCUseGasGainCorrections = cms.bool( False ),
    XTasymmetry_ME31 = cms.double( 0.0 ),
    readBadChambers = cms.bool( True ),
    NoiseLevel_ME13 = cms.double( 8.0 ),
    NoiseLevel_ME12 = cms.double( 9.0 ),
    NoiseLevel_ME32 = cms.double( 9.0 ),
    NoiseLevel_ME31 = cms.double( 9.0 ),
    XTasymmetry_ME32 = cms.double( 0.0 ),
    ConstSyst_ME41 = cms.double( 0.0 ),
    CSCStripClusterSize = cms.untracked.int32( 3 ),
    CSCStripClusterChargeCut = cms.double( 25.0 ),
    CSCStripPeakThreshold = cms.double( 10.0 ),
    readBadChannels = cms.bool( True ),
    UseParabolaFit = cms.bool( False ),
    XTasymmetry_ME13 = cms.double( 0.0 ),
    XTasymmetry_ME12 = cms.double( 0.0 ),
    wireDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCWireDigi' ),
    ConstSyst_ME12 = cms.double( 0.0 ),
    ConstSyst_ME13 = cms.double( 0.0 ),
    ConstSyst_ME32 = cms.double( 0.0 ),
    ConstSyst_ME31 = cms.double( 0.0 ),
    UseAverageTime = cms.bool( False ),
    NoiseLevel_ME1a = cms.double( 7.0 ),
    NoiseLevel_ME1b = cms.double( 8.0 ),
    CSCWireClusterDeltaT = cms.int32( 1 ),
    CSCUseStaticPedestals = cms.bool( False ),
    stripDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCStripDigi' ),
    CSCstripWireDeltaTime = cms.int32( 8 ),
    NoiseLevel_ME21 = cms.double( 9.0 ),
    NoiseLevel_ME22 = cms.double( 9.0 ),
    NoiseLevel_ME41 = cms.double( 9.0 )
)
process.hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
    algo_psets = cms.VPSet( 
      cms.PSet(  chamber_types = cms.vstring( 'ME1/a',
  'ME1/b',
  'ME1/2',
  'ME1/3',
  'ME2/1',
  'ME2/2',
  'ME3/1',
  'ME3/2',
  'ME4/1',
  'ME4/2' ),
        algo_name = cms.string( "CSCSegAlgoST" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 ),
        algo_psets = cms.VPSet( 
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.double( 1.5 ),
            maxRecHitsInCluster = cms.int32( 20 ),
            dPhiFineMax = cms.double( 0.025 ),
            preClusteringUseChaining = cms.bool( True ),
            ForceCovariance = cms.bool( False ),
            hitDropLimit6Hits = cms.double( 0.3333 ),
            NormChi2Cut2D = cms.double( 20.0 ),
            BPMinImprovement = cms.double( 10000.0 ),
            Covariance = cms.double( 0.0 ),
            tanPhiMax = cms.double( 0.5 ),
            SeedBig = cms.double( 0.0015 ),
            onlyBestSegment = cms.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            SeedSmall = cms.double( 2.0E-4 ),
            curvePenalty = cms.double( 2.0 ),
            dXclusBoxMax = cms.double( 4.0 ),
            BrutePruning = cms.bool( True ),
            curvePenaltyThreshold = cms.double( 0.85 ),
            CorrectTheErrors = cms.bool( True ),
            hitDropLimit4Hits = cms.double( 0.6 ),
            useShowering = cms.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            tanThetaMax = cms.double( 1.2 ),
            NormChi2Cut3D = cms.double( 10.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            ForceCovarianceAll = cms.bool( False ),
            yweightPenaltyThreshold = cms.double( 1.0 ),
            prePrunLimit = cms.double( 3.17 ),
            hitDropLimit5Hits = cms.double( 0.8 ),
            preClustering = cms.bool( True ),
            prePrun = cms.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.bool( True ),
            dYclusBoxMax = cms.double( 8.0 )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.double( 1.5 ),
            maxRecHitsInCluster = cms.int32( 24 ),
            dPhiFineMax = cms.double( 0.025 ),
            preClusteringUseChaining = cms.bool( True ),
            ForceCovariance = cms.bool( False ),
            hitDropLimit6Hits = cms.double( 0.3333 ),
            NormChi2Cut2D = cms.double( 20.0 ),
            BPMinImprovement = cms.double( 10000.0 ),
            Covariance = cms.double( 0.0 ),
            tanPhiMax = cms.double( 0.5 ),
            SeedBig = cms.double( 0.0015 ),
            onlyBestSegment = cms.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            SeedSmall = cms.double( 2.0E-4 ),
            curvePenalty = cms.double( 2.0 ),
            dXclusBoxMax = cms.double( 4.0 ),
            BrutePruning = cms.bool( True ),
            curvePenaltyThreshold = cms.double( 0.85 ),
            CorrectTheErrors = cms.bool( True ),
            hitDropLimit4Hits = cms.double( 0.6 ),
            useShowering = cms.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            tanThetaMax = cms.double( 1.2 ),
            NormChi2Cut3D = cms.double( 10.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            ForceCovarianceAll = cms.bool( False ),
            yweightPenaltyThreshold = cms.double( 1.0 ),
            prePrunLimit = cms.double( 3.17 ),
            hitDropLimit5Hits = cms.double( 0.8 ),
            preClustering = cms.bool( True ),
            prePrun = cms.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.bool( True ),
            dYclusBoxMax = cms.double( 8.0 )
          )
        )
      )
    ),
    algo_type = cms.int32( 1 )
)
process.hltMuonRPCDigis = cms.EDProducer( "RPCUnpackingModule",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    doSynchro = cms.bool( False )
)
process.hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" ),
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    deadSource = cms.string( "File" ),
    maskSource = cms.string( "File" )
)
process.hltL2OfflineMuonSeeds = cms.EDProducer( "MuonSeedGenerator",
    SMB_21 = cms.vdouble( 1.043, -0.124, 0.0, 0.183, 0.0, 0.0 ),
    SMB_20 = cms.vdouble( 1.011, -0.052, 0.0, 0.188, 0.0, 0.0 ),
    SMB_22 = cms.vdouble( 1.474, -0.758, 0.0, 0.185, 0.0, 0.0 ),
    OL_2213 = cms.vdouble( 0.117, 0.0, 0.0, 0.044, 0.0, 0.0 ),
    SME_11 = cms.vdouble( 3.295, -1.527, 0.112, 0.378, 0.02, 0.0 ),
    SME_13 = cms.vdouble( -1.286, 1.711, 0.0, 0.356, 0.0, 0.0 ),
    SME_12 = cms.vdouble( 0.102, 0.599, 0.0, 0.38, 0.0, 0.0 ),
    DT_34_2_scale = cms.vdouble( -11.901897, 0.0 ),
    OL_1213_0_scale = cms.vdouble( -4.488158, 0.0 ),
    OL_1222_0_scale = cms.vdouble( -5.810449, 0.0 ),
    DT_13 = cms.vdouble( 0.315, 0.068, -0.127, 0.051, -0.002, 0.0 ),
    DT_12 = cms.vdouble( 0.183, 0.054, -0.087, 0.028, 0.002, 0.0 ),
    DT_14 = cms.vdouble( 0.359, 0.052, -0.107, 0.072, -0.004, 0.0 ),
    CSC_13_3_scale = cms.vdouble( -1.701268, 0.0 ),
    DT_24_2_scale = cms.vdouble( -6.63094, 0.0 ),
    CSC_23 = cms.vdouble( -0.081, 0.113, -0.029, 0.015, 0.008, 0.0 ),
    CSC_24 = cms.vdouble( 0.004, 0.021, -0.002, 0.053, 0.0, 0.0 ),
    OL_2222 = cms.vdouble( 0.107, 0.0, 0.0, 0.04, 0.0, 0.0 ),
    DT_14_2_scale = cms.vdouble( -4.808546, 0.0 ),
    SMB_10 = cms.vdouble( 1.387, -0.038, 0.0, 0.19, 0.0, 0.0 ),
    SMB_11 = cms.vdouble( 1.247, 0.72, -0.802, 0.229, -0.075, 0.0 ),
    SMB_12 = cms.vdouble( 2.128, -0.956, 0.0, 0.199, 0.0, 0.0 ),
    SME_21 = cms.vdouble( -0.529, 1.194, -0.358, 0.472, 0.086, 0.0 ),
    SME_22 = cms.vdouble( -1.207, 1.491, -0.251, 0.189, 0.243, 0.0 ),
    DT_13_2_scale = cms.vdouble( -4.257687, 0.0 ),
    CSC_34 = cms.vdouble( 0.062, -0.067, 0.019, 0.021, 0.003, 0.0 ),
    SME_22_0_scale = cms.vdouble( -3.457901, 0.0 ),
    DT_24_1_scale = cms.vdouble( -7.490909, 0.0 ),
    OL_1232_0_scale = cms.vdouble( -5.964634, 0.0 ),
    DT_23_1_scale = cms.vdouble( -5.320346, 0.0 ),
    SME_13_0_scale = cms.vdouble( 0.104905, 0.0 ),
    SMB_22_0_scale = cms.vdouble( 1.346681, 0.0 ),
    CSC_12_1_scale = cms.vdouble( -6.434242, 0.0 ),
    DT_34 = cms.vdouble( 0.044, 0.004, -0.013, 0.029, 0.003, 0.0 ),
    SME_32 = cms.vdouble( -0.901, 1.333, -0.47, 0.41, 0.073, 0.0 ),
    SME_31 = cms.vdouble( -1.594, 1.482, -0.317, 0.487, 0.097, 0.0 ),
    CSC_13_2_scale = cms.vdouble( -6.077936, 0.0 ),
    crackEtas = cms.vdouble( 0.2, 1.6, 1.7 ),
    SME_11_0_scale = cms.vdouble( 1.325085, 0.0 ),
    SMB_20_0_scale = cms.vdouble( 1.486168, 0.0 ),
    DT_13_1_scale = cms.vdouble( -4.520923, 0.0 ),
    CSC_24_1_scale = cms.vdouble( -6.055701, 0.0 ),
    CSC_01_1_scale = cms.vdouble( -1.915329, 0.0 ),
    DT_23 = cms.vdouble( 0.13, 0.023, -0.057, 0.028, 0.004, 0.0 ),
    DT_24 = cms.vdouble( 0.176, 0.014, -0.051, 0.051, 0.003, 0.0 ),
    SMB_12_0_scale = cms.vdouble( 2.283221, 0.0 ),
    SMB_30_0_scale = cms.vdouble( -3.629838, 0.0 ),
    SME_42 = cms.vdouble( -0.003, 0.005, 0.005, 0.608, 0.076, 0.0 ),
    SME_41 = cms.vdouble( -0.003, 0.005, 0.005, 0.608, 0.076, 0.0 ),
    CSC_12_2_scale = cms.vdouble( -1.63622, 0.0 ),
    DT_34_1_scale = cms.vdouble( -13.783765, 0.0 ),
    CSC_34_1_scale = cms.vdouble( -11.520507, 0.0 ),
    OL_2213_0_scale = cms.vdouble( -7.239789, 0.0 ),
    SMB_32_0_scale = cms.vdouble( -3.054156, 0.0 ),
    CSC_12_3_scale = cms.vdouble( -1.63622, 0.0 ),
    SME_21_0_scale = cms.vdouble( -0.040862, 0.0 ),
    OL_1232 = cms.vdouble( 0.184, 0.0, 0.0, 0.066, 0.0, 0.0 ),
    DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
    SMB_10_0_scale = cms.vdouble( 2.448566, 0.0 ),
    EnableDTMeasurement = cms.bool( True ),
    CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
    CSC_23_2_scale = cms.vdouble( -6.079917, 0.0 ),
    scaleDT = cms.bool( True ),
    DT_12_2_scale = cms.vdouble( -3.518165, 0.0 ),
    OL_1222 = cms.vdouble( 0.848, -0.591, 0.0, 0.062, 0.0, 0.0 ),
    CSC_23_1_scale = cms.vdouble( -19.084285, 0.0 ),
    OL_1213 = cms.vdouble( 0.96, -0.737, 0.0, 0.052, 0.0, 0.0 ),
    CSC_02 = cms.vdouble( 0.612, -0.207, 0.0, 0.067, -0.001, 0.0 ),
    CSC_03 = cms.vdouble( 0.787, -0.338, 0.029, 0.101, -0.008, 0.0 ),
    CSC_01 = cms.vdouble( 0.166, 0.0, 0.0, 0.031, 0.0, 0.0 ),
    SMB_32 = cms.vdouble( 0.67, -0.327, 0.0, 0.22, 0.0, 0.0 ),
    SMB_30 = cms.vdouble( 0.505, -0.022, 0.0, 0.215, 0.0, 0.0 ),
    SMB_31 = cms.vdouble( 0.549, -0.145, 0.0, 0.207, 0.0, 0.0 ),
    crackWindow = cms.double( 0.04 ),
    CSC_14_3_scale = cms.vdouble( -1.969563, 0.0 ),
    SMB_31_0_scale = cms.vdouble( -3.323768, 0.0 ),
    DT_12_1_scale = cms.vdouble( -3.692398, 0.0 ),
    SMB_21_0_scale = cms.vdouble( 1.58384, 0.0 ),
    DT_23_2_scale = cms.vdouble( -5.117625, 0.0 ),
    SME_12_0_scale = cms.vdouble( 2.279181, 0.0 ),
    DT_14_1_scale = cms.vdouble( -5.644816, 0.0 ),
    beamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    SMB_11_0_scale = cms.vdouble( 2.56363, 0.0 ),
    EnableCSCMeasurement = cms.bool( True ),
    CSC_14 = cms.vdouble( 0.606, -0.181, -0.002, 0.111, -0.003, 0.0 ),
    OL_2222_0_scale = cms.vdouble( -7.667231, 0.0 ),
    CSC_13 = cms.vdouble( 0.901, -1.302, 0.533, 0.045, 0.005, 0.0 ),
    CSC_12 = cms.vdouble( -0.161, 0.254, -0.047, 0.042, -0.007, 0.0 )
)
process.hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGenerator",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    InputObjects = cms.InputTag( "hltL1extraParticles" ),
    L1MaxEta = cms.double( 2.5 ),
    OfflineSeedLabel = cms.untracked.InputTag( "hltL2OfflineMuonSeeds" ),
    L1MinPt = cms.double( 0.0 ),
    L1MinQuality = cms.uint32( 1 ),
    GMTReadoutCollection = cms.InputTag( "hltGtDigis" ),
    UseOfflineSeed = cms.untracked.bool( True ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" )
)
process.hltL2Muons = cms.EDProducer( "L2MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny',
        'hltESPFastSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    SeedTransformerParameters = cms.PSet( 
      Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      NMinRecHits = cms.uint32( 2 ),
      UseSubRecHits = cms.bool( False ),
      Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      RescaleError = cms.double( 100.0 )
    ),
    L2TrajBuilderParameters = cms.PSet( 
      DoRefit = cms.bool( False ),
      SeedPropagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      FilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        FitDirection = cms.string( "insideOut" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 1000.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          Granularity = cms.int32( 0 ),
          ExcludeRPCFromFit = cms.bool( False ),
          UseInvalidHits = cms.bool( True ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      ),
      NavigationType = cms.string( "Standard" ),
      SeedTransformerParameters = cms.PSet( 
        Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        NMinRecHits = cms.uint32( 2 ),
        UseSubRecHits = cms.bool( False ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        RescaleError = cms.double( 100.0 )
      ),
      DoBackwardFilter = cms.bool( True ),
      SeedPosition = cms.string( "in" ),
      BWFilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        FitDirection = cms.string( "outsideIn" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 100.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          Granularity = cms.int32( 2 ),
          ExcludeRPCFromFit = cms.bool( False ),
          UseInvalidHits = cms.bool( True ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        BWSeedType = cms.string( "fromGenerator" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      ),
      DoSeedRefit = cms.bool( False )
    ),
    DoSeedRefit = cms.bool( False ),
    TrackLoaderParameters = cms.PSet( 
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      DoSmoothing = cms.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( True )
    )
)
process.hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltL2fL1sMu7L1fEta2p1L2FilteredEta2p1Filtered7 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMu7L1FilteredEta2p1Filtered0" ),
    MinPt = cms.double( 7.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.1 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltSiPixelDigis = cms.EDProducer( "SiPixelRawToDigi",
    UseQualityInfo = cms.bool( False ),
    CheckPixelOrder = cms.bool( False ),
    IncludeErrors = cms.bool( False ),
    UseCablingTree = cms.untracked.bool( True ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    ErrorList = cms.vint32(  ),
    Regions = cms.PSet(  ),
    Timing = cms.untracked.bool( False ),
    UserErrorList = cms.vint32(  )
)
process.hltSiPixelClusters = cms.EDProducer( "SiPixelClusterProducer",
    src = cms.InputTag( "hltSiPixelDigis" ),
    ChannelThreshold = cms.int32( 1000 ),
    maxNumberOfClusters = cms.int32( 20000 ),
    VCaltoElectronGain = cms.int32( 65 ),
    MissCalibrate = cms.untracked.bool( True ),
    SplitClusters = cms.bool( False ),
    VCaltoElectronOffset = cms.int32( -414 ),
    payloadType = cms.string( "HLT" ),
    SeedThreshold = cms.int32( 1000 ),
    ClusterThreshold = cms.double( 4000.0 )
)
process.hltSiPixelRecHits = cms.EDProducer( "SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32( 0 ),
    src = cms.InputTag( "hltSiPixelClusters" ),
    CPE = cms.string( "hltESPPixelCPEGeneric" )
)
process.hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)
process.hltSiStripRawToClustersFacility = cms.EDProducer( "SiStripRawToClusters",
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    DoAPVEmulatorCheck = cms.bool( False ),
    Algorithms = cms.PSet( 
      SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
      CommonModeNoiseSubtractionMode = cms.string( "Median" ),
      PedestalSubtractionFedMode = cms.bool( True ),
      TruncateInSuppressor = cms.bool( True ),
      doAPVRestore = cms.bool( False ),
      useCMMeanMap = cms.bool( False )
    ),
    Clusterizer = cms.PSet( 
      ChannelThreshold = cms.double( 2.0 ),
      MaxSequentialBad = cms.uint32( 1 ),
      MaxSequentialHoles = cms.uint32( 0 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      MaxAdjacentBad = cms.uint32( 0 ),
      QualityLabel = cms.string( "" ),
      SeedThreshold = cms.double( 3.0 ),
      ClusterThreshold = cms.double( 5.0 ),
      setDetId = cms.bool( True ),
      RemoveApvShots = cms.bool( True )
    )
)
process.hltSiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltESPMeasurementTracker" )
)
process.hltL3TrajSeedOIState = cms.EDProducer( "TSGFromL2Muon",
    TkSeedGenerator = cms.PSet( 
      propagatorCompatibleName = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
      option = cms.uint32( 3 ),
      maxChi2 = cms.double( 40.0 ),
      errorMatrixPset = cms.PSet( 
        atIP = cms.bool( True ),
        action = cms.string( "use" ),
        errorMatrixValuesPSet = cms.PSet( 
          pf3_V12 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V13 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V11 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V14 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V15 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
          pf3_V33 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          zAxis = cms.vdouble( -3.14159, 3.14159 ),
          pf3_V44 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
          pf3_V22 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V23 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V45 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V55 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V34 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V35 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V25 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V24 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          )
        )
      ),
      propagatorName = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
      manySeeds = cms.bool( False ),
      copyMuonRecHit = cms.bool( False ),
      ComponentName = cms.string( "TSGForRoadSearch" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSteppingHelixPropagatorOpposite',
        'hltESPSteppingHelixPropagatorAlong' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet(  ),
    PtCut = cms.double( 1.0 )
)
process.hltL3TrackCandidateFromL2OIState = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltL3TrajSeedOIState" ),
    reverseTrajectories = cms.bool( True ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPMuonCkfTrajectoryBuilderSeedHit" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltL3TkTracksFromL2OIState = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltL3TrackCandidateFromL2OIState" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltL3MuonsOIState = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      MuonTrackingRegionBuilder = cms.PSet( 
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        OnDemand = cms.double( -1.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Rescale_phi = cms.double( 3.0 ),
        Eta_fixed = cms.double( 0.2 ),
        DeltaZ_Region = cms.double( 15.9 ),
        MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Eta_min = cms.double( 0.05 ),
        Phi_fixed = cms.double( 0.2 ),
        DeltaR = cms.double( 0.2 ),
        EscapePt = cms.double( 1.5 ),
        UseFixedRegion = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        Rescale_eta = cms.double( 3.0 ),
        Phi_min = cms.double( 0.05 ),
        UseVertex = cms.bool( False ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.001 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltL3TkTracksFromL2OIState" )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltL3TrajSeedOIHit = cms.EDProducer( "TSGFromL2Muon",
    TkSeedGenerator = cms.PSet( 
      PSetNames = cms.vstring( 'skipTSG',
        'iterativeTSG' ),
      L3TkCollectionA = cms.InputTag( "hltL3MuonsOIState" ),
      iterativeTSG = cms.PSet( 
        ErrorRescaling = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "unused" ),
        MaxChi2 = cms.double( 40.0 ),
        errorMatrixPset = cms.PSet( 
          atIP = cms.bool( True ),
          action = cms.string( "use" ),
          errorMatrixValuesPSet = cms.PSet( 
            pf3_V12 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V13 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V11 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V14 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V15 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
            pf3_V33 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            zAxis = cms.vdouble( -3.14159, 3.14159 ),
            pf3_V44 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
            pf3_V22 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V23 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V45 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V55 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V34 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V35 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V25 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V24 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            )
          )
        ),
        UpdateState = cms.bool( True ),
        MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
        SelectState = cms.bool( False ),
        SigmaZ = cms.double( 25.0 ),
        ResetMethod = cms.string( "matrix" ),
        ComponentName = cms.string( "TSGFromPropagation" ),
        UseVertexState = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" )
      ),
      skipTSG = cms.PSet(  ),
      ComponentName = cms.string( "DualByL2TSG" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'PropagatorWithMaterial',
        'hltESPSmartPropagatorAnyOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet( 
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      directionCleaner = cms.bool( True )
    ),
    PtCut = cms.double( 1.0 )
)
process.hltL3TrackCandidateFromL2OIHit = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltL3TrajSeedOIHit" ),
    reverseTrajectories = cms.bool( True ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPMuonCkfTrajectoryBuilder" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltL3TkTracksFromL2OIHit = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltL3TrackCandidateFromL2OIHit" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltL3MuonsOIHit = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      MuonTrackingRegionBuilder = cms.PSet( 
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        OnDemand = cms.double( -1.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Rescale_phi = cms.double( 3.0 ),
        Eta_fixed = cms.double( 0.2 ),
        DeltaZ_Region = cms.double( 15.9 ),
        MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Eta_min = cms.double( 0.05 ),
        Phi_fixed = cms.double( 0.2 ),
        DeltaR = cms.double( 0.2 ),
        EscapePt = cms.double( 1.5 ),
        UseFixedRegion = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        Rescale_eta = cms.double( 3.0 ),
        Phi_min = cms.double( 0.05 ),
        UseVertex = cms.bool( False ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.001 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltL3TkTracksFromL2OIHit" )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltL3TkFromL2OICombination = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltL3MuonsOIState','hltL3MuonsOIHit' )
)
process.hltL3TrajSeedIOHit = cms.EDProducer( "TSGFromL2Muon",
    TkSeedGenerator = cms.PSet( 
      PSetNames = cms.vstring( 'skipTSG',
        'iterativeTSG' ),
      L3TkCollectionA = cms.InputTag( "hltL3TkFromL2OICombination" ),
      iterativeTSG = cms.PSet( 
        firstTSG = cms.PSet( 
          ComponentName = cms.string( "TSGFromOrderedHits" ),
          OrderedHitsFactoryPSet = cms.PSet( 
            ComponentName = cms.string( "StandardHitTripletGenerator" ),
            GeneratorPSet = cms.PSet( 
              useBending = cms.bool( True ),
              useFixedPreFiltering = cms.bool( False ),
              maxElement = cms.uint32( 0 ),
              phiPreFiltering = cms.double( 0.3 ),
              extraHitRPhitolerance = cms.double( 0.06 ),
              useMultScattering = cms.bool( True ),
              ComponentName = cms.string( "PixelTripletHLTGenerator" ),
              extraHitRZtolerance = cms.double( 0.06 ),
              SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
            ),
            SeedingLayers = cms.string( "hltESPPixelLayerTriplets" )
          ),
          TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
        ),
        PSetNames = cms.vstring( 'firstTSG',
          'secondTSG' ),
        ComponentName = cms.string( "CombinedTSG" ),
        thirdTSG = cms.PSet( 
          PSetNames = cms.vstring( 'endcapTSG',
            'barrelTSG' ),
          barrelTSG = cms.PSet(  ),
          endcapTSG = cms.PSet( 
            ComponentName = cms.string( "TSGFromOrderedHits" ),
            OrderedHitsFactoryPSet = cms.PSet( 
              maxElement = cms.uint32( 0 ),
              ComponentName = cms.string( "StandardHitPairGenerator" ),
              SeedingLayers = cms.string( "hltESPMixedLayerPairs" ),
              useOnDemandTracker = cms.untracked.int32( 0 )
            ),
            TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
          ),
          etaSeparation = cms.double( 2.0 ),
          ComponentName = cms.string( "DualByEtaTSG" )
        ),
        secondTSG = cms.PSet( 
          ComponentName = cms.string( "TSGFromOrderedHits" ),
          OrderedHitsFactoryPSet = cms.PSet( 
            maxElement = cms.uint32( 0 ),
            ComponentName = cms.string( "StandardHitPairGenerator" ),
            SeedingLayers = cms.string( "hltESPPixelLayerPairs" ),
            useOnDemandTracker = cms.untracked.int32( 0 )
          ),
          TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
        )
      ),
      skipTSG = cms.PSet(  ),
      ComponentName = cms.string( "DualByL2TSG" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'PropagatorWithMaterial' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    MuonTrackingRegionBuilder = cms.PSet( 
      EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
      EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
      OnDemand = cms.double( -1.0 ),
      Rescale_Dz = cms.double( 3.0 ),
      vertexCollection = cms.InputTag( "pixelVertices" ),
      Rescale_phi = cms.double( 3.0 ),
      Eta_fixed = cms.double( 0.2 ),
      DeltaZ_Region = cms.double( 15.9 ),
      MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
      PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
      Eta_min = cms.double( 0.1 ),
      Phi_fixed = cms.double( 0.2 ),
      DeltaR = cms.double( 0.2 ),
      EscapePt = cms.double( 1.5 ),
      UseFixedRegion = cms.bool( False ),
      PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
      Rescale_eta = cms.double( 3.0 ),
      Phi_min = cms.double( 0.1 ),
      UseVertex = cms.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
    ),
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet( 
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      directionCleaner = cms.bool( True )
    ),
    PtCut = cms.double( 1.0 )
)
process.hltL3TrackCandidateFromL2IOHit = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltL3TrajSeedIOHit" ),
    reverseTrajectories = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPMuonCkfTrajectoryBuilder" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltL3TkTracksFromL2IOHit = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltL3TrackCandidateFromL2IOHit" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltL3MuonsIOHit = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      MuonTrackingRegionBuilder = cms.PSet( 
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        OnDemand = cms.double( -1.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Rescale_phi = cms.double( 3.0 ),
        Eta_fixed = cms.double( 0.2 ),
        DeltaZ_Region = cms.double( 15.9 ),
        MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Eta_min = cms.double( 0.05 ),
        Phi_fixed = cms.double( 0.2 ),
        DeltaR = cms.double( 0.2 ),
        EscapePt = cms.double( 1.5 ),
        UseFixedRegion = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        Rescale_eta = cms.double( 3.0 ),
        Phi_min = cms.double( 0.05 ),
        UseVertex = cms.bool( False ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.001 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltL3TkTracksFromL2IOHit" )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltL3TrajectorySeed = cms.EDProducer( "L3MuonTrajectorySeedCombiner",
    labels = cms.VInputTag( 'hltL3TrajSeedIOHit','hltL3TrajSeedOIState','hltL3TrajSeedOIHit' )
)
process.hltL3TrackCandidateFromL2 = cms.EDProducer( "L3TrackCandCombiner",
    labels = cms.VInputTag( 'hltL3TrackCandidateFromL2IOHit','hltL3TrackCandidateFromL2OIHit','hltL3TrackCandidateFromL2OIState' )
)
process.hltL3TkTracksFromL2 = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltL3TkTracksFromL2IOHit','hltL3TkTracksFromL2OIHit','hltL3TkTracksFromL2OIState' )
)
process.hltL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltL3MuonsOIState','hltL3MuonsOIHit','hltL3MuonsIOHit' )
)
process.hltL3Muons = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltL3MuonsOIState','hltL3MuonsOIHit','hltL3MuonsIOHit' )
)
process.hltL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputLinksObjects = cms.InputTag( "hltL3MuonsLinksCombination" ),
    InputObjects = cms.InputTag( "hltL3Muons" ),
    MuonPtOption = cms.string( "Tracker" )
)
process.hltL3fL1sMu7L1fEta2p1L2fEta2p1f7L3FilteredEta2p1Filtered15 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL2fL1sMu7L1fEta2p1L2FilteredEta2p1Filtered7" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.1 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 15.0 )
)
process.hltL1sL1SingleEG12 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG12" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreEle22CaloIdLCaloIsoVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalRegionalEgammaFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet( 
    ),
    sourceTag_es = cms.InputTag( "hltESRawToRecHitFacility" ),
    doES = cms.bool( True ),
    type = cms.string( "egamma" ),
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    EmJobPSet = cms.VPSet( 
      cms.PSet(  regionEtaMargin = cms.double( 0.25 ),
        regionPhiMargin = cms.double( 0.4 ),
        Ptmin = cms.double( 5.0 ),
        Source = cms.InputTag( 'hltL1extraParticles','Isolated' )
      ),
      cms.PSet(  regionEtaMargin = cms.double( 0.25 ),
        regionPhiMargin = cms.double( 0.4 ),
        Ptmin = cms.double( 5.0 ),
        Source = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
      )
    ),
    CandJobPSet = cms.VPSet( 
    ),
    MuonJobPSet = cms.PSet(  ),
    esInstance = cms.untracked.string( "es" ),
    MuJobPSet = cms.PSet(  )
)
process.hltEcalRegionalEgammaRecHit = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( True ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    sourceTag = cms.InputTag( "hltEcalRegionalEgammaFEDs" ),
    cleaningConfig = cms.PSet( 
      e6e2thresh = cms.double( 0.04 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      tightenCrack_e1_double = cms.double( 2.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      ignoreOutOfTimeThresh = cms.double( 1.0E9 ),
      cThreshold_endcap = cms.double( 15.0 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      cThreshold_double = cms.double( 10.0 )
    ),
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" )
)
process.hltESRegionalEgammaRecHit = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( False ),
    rechitCollection = cms.string( "EcalRecHitsES" ),
    EErechitCollection = cms.string( "" ),
    EBrechitCollection = cms.string( "" ),
    sourceTag = cms.InputTag( 'hltEcalRegionalEgammaFEDs','es' ),
    cleaningConfig = cms.PSet(  ),
    lazyGetterTag = cms.InputTag( "hltESRawToRecHitFacility" )
)
process.hltHybridSuperClustersL1Seeded = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    xi = cms.double( 0.0 ),
    regionEtaMargin = cms.double( 0.14 ),
    regionPhiMargin = cms.double( 0.4 ),
    severityRecHitThreshold = cms.double( 4.0 ),
    RecHitFlagToBeExcluded = cms.vstring(  ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    eThreshA = cms.double( 0.003 ),
    basicclusterCollection = cms.string( "" ),
    eThreshB = cms.double( 0.1 ),
    dynamicPhiRoad = cms.bool( False ),
    RecHitSeverityToBeExcluded = cms.vstring( 'kWeird' ),
    l1UpperThr = cms.double( 999.0 ),
    excludeFlagged = cms.bool( True ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    ),
    l1LowerThr = cms.double( 5.0 ),
    doIsolated = cms.bool( True ),
    eseed = cms.double( 0.35 ),
    ethresh = cms.double( 0.1 ),
    ewing = cms.double( 0.0 ),
    useEtForXi = cms.bool( True ),
    step = cms.int32( 17 ),
    debugLevel = cms.string( "INFO" ),
    dynamicEThresh = cms.bool( False ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    superclusterCollection = cms.string( "" ),
    HybridBarrelSeedThr = cms.double( 1.5 ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1LowerThrIgnoreIsolation = cms.double( 0.0 ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" )
)
process.hltCorrectedHybridSuperClustersL1Seeded = cms.EDProducer( "EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    superClusterAlgo = cms.string( "Hybrid" ),
    etThresh = cms.double( 1.0 ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1Seeded" ),
    applyEnergyCorrection = cms.bool( True ),
    isl_fCorrPset = cms.PSet(  ),
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    fix_fCorrPset = cms.PSet(  ),
    modeEE = cms.int32( 0 ),
    modeEB = cms.int32( 0 ),
    dyn_fCorrPset = cms.PSet(  ),
    energyCorrectorName = cms.string( "EcalClusterEnergyCorrectionObjectSpecific" ),
    applyCrackCorrection = cms.bool( False ),
    hyb_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 1.1 ),
      fBremVec = cms.vdouble( -0.05208, 0.1331, 0.9196, -5.735E-4, 1.343 ),
      brLinearHighThr = cms.double( 8.0 ),
      fEtEtaVec = cms.vdouble( 1.0012, -0.5714, 0.0, 0.0, 0.0, 0.5549, 12.74, 1.0448, 0.0, 0.0, 0.0, 0.0, 8.0, 1.023, -0.00181, 0.0, 0.0 )
    )
)
process.hltMulti5x5BasicClustersL1Seeded = cms.EDProducer( "EgammaHLTMulti5x5ClusterProducer",
    l1LowerThr = cms.double( 5.0 ),
    Multi5x5BarrelSeedThr = cms.double( 0.5 ),
    Multi5x5EndcapSeedThr = cms.double( 0.18 ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "notused" ),
    doEndcaps = cms.bool( True ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    RecHitFlagToBeExcluded = cms.vstring(  ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    ),
    VerbosityLevel = cms.string( "ERROR" ),
    doIsolated = cms.bool( True ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    l1LowerThrIgnoreIsolation = cms.double( 0.0 ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    doBarrel = cms.bool( False ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    l1UpperThr = cms.double( 999.0 )
)
process.hltMulti5x5SuperClustersL1Seeded = cms.EDProducer( "Multi5x5SuperClusterProducer",
    barrelSuperclusterCollection = cms.string( "multi5x5BarrelSuperClusters" ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    barrelClusterCollection = cms.string( "multi5x5BarrelBasicClusters" ),
    dynamicPhiRoad = cms.bool( False ),
    endcapClusterProducer = cms.string( "hltMulti5x5BasicClustersL1Seeded" ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    barrelClusterProducer = cms.string( "notused" ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    endcapSuperclusterCollection = cms.string( "multi5x5EndcapSuperClusters" ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    bremRecoveryPset = cms.PSet( 
      barrel = cms.PSet(  ),
      endcap = cms.PSet( 
        a = cms.double( 47.85 ),
        c = cms.double( 0.1201 ),
        b = cms.double( 108.8 )
      ),
      doEndcaps = cms.bool( True ),
      doBarrel = cms.bool( False )
    ),
    doEndcaps = cms.bool( True ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    doBarrel = cms.bool( False )
)
process.hltMulti5x5EndcapSuperClustersWithPreshowerL1Seeded = cms.EDProducer( "PreshowerClusterProducer",
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    etThresh = cms.double( 5.0 ),
    preshRecHitProducer = cms.InputTag( 'hltESRegionalEgammaRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( 'hltMulti5x5SuperClustersL1Seeded','multi5x5EndcapSuperClusters' ),
    preshNclust = cms.int32( 4 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 )
)
process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Seeded = cms.EDProducer( "EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    superClusterAlgo = cms.string( "Multi5x5" ),
    etThresh = cms.double( 1.0 ),
    rawSuperClusterProducer = cms.InputTag( "hltMulti5x5EndcapSuperClustersWithPreshowerL1Seeded" ),
    applyEnergyCorrection = cms.bool( True ),
    isl_fCorrPset = cms.PSet(  ),
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    fix_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.6 ),
      fBremVec = cms.vdouble( -0.04163, 0.08552, 0.95048, -0.002308, 1.077 ),
      brLinearHighThr = cms.double( 6.0 ),
      fEtEtaVec = cms.vdouble( 0.9746, -6.512, 0.0, 0.0, 0.02771, 4.983, 0.0, 0.0, -0.007288, -0.9446, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 )
    ),
    modeEE = cms.int32( 0 ),
    modeEB = cms.int32( 0 ),
    dyn_fCorrPset = cms.PSet(  ),
    energyCorrectorName = cms.string( "EcalClusterEnergyCorrectionObjectSpecific" ),
    applyCrackCorrection = cms.bool( False ),
    hyb_fCorrPset = cms.PSet(  )
)
process.hltL1SeededRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Seeded" ),
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1Seeded" ),
    recoEcalCandidateCollection = cms.string( "" )
)
process.hltEGRegionalL1SingleEG12 = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    saveTags = cms.bool( False ),
    endcap_end = cms.double( 2.65 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    barrel_end = cms.double( 1.4791 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candIsolatedTag = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    region_phi_size = cms.double( 1.044 ),
    region_eta_size = cms.double( 0.522 ),
    L1SeedFilterTag = cms.InputTag( "hltL1sL1SingleEG12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    candNonIsolatedTag = cms.InputTag( "" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
)
process.hltEG22EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG12" ),
    etcutEB = cms.double( 22.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 22.0 )
)
process.hltL1SeededHLTClusterShape = cms.EDProducer( "EgammaHLTClusterShapeProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    isIeta = cms.bool( True )
)
process.hltEG22CaloIdLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEG22EtFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltL1SeededPhotonEcalIso = cms.EDProducer( "EgammaHLTEcalRecIsolationProducer",
    etMinEndcap = cms.double( 0.11 ),
    tryBoth = cms.bool( True ),
    ecalBarrelRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    rhoMax = cms.double( 9.9999999E7 ),
    useNumCrystals = cms.bool( True ),
    etMinBarrel = cms.double( -9999.0 ),
    doRhoCorrection = cms.bool( False ),
    eMinEndcap = cms.double( -9999.0 ),
    intRadiusEndcap = cms.double( 3.0 ),
    jurassicWidth = cms.double( 3.0 ),
    useIsolEt = cms.bool( True ),
    ecalBarrelRecHitCollection = cms.InputTag( "EcalRecHitsEB" ),
    recoEcalCandidateProducer = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    eMinBarrel = cms.double( 0.095 ),
    effectiveAreaEndcap = cms.double( 0.046 ),
    ecalEndcapRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    extRadius = cms.double( 0.3 ),
    intRadiusBarrel = cms.double( 3.0 ),
    subtract = cms.bool( False ),
    rhoScale = cms.double( 1.0 ),
    effectiveAreaBarrel = cms.double( 0.101 ),
    ecalEndcapRecHitCollection = cms.InputTag( "EcalRecHitsEE" ),
    rhoProducer = cms.InputTag( 'hltKT6CaloJets','rho' )
)
process.hltEG22CaloIdLCaloIsoVLEcalIsoFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.2 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.2 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonEcalIso" ),
    candTag = cms.InputTag( "hltEG22CaloIdLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
    UnpackZDC = cms.untracked.bool( True ),
    FilterDataQuality = cms.bool( True ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    ComplainEmptyData = cms.untracked.bool( False ),
    UnpackCalib = cms.untracked.bool( True ),
    UnpackTTP = cms.untracked.bool( False ),
    lastSample = cms.int32( 9 ),
    firstSample = cms.int32( 0 )
)
process.hltHbhereco = cms.EDProducer( "HcalHitReconstructor",
    digiTimeFromDB = cms.bool( True ),
    S9S1stat = cms.PSet(  ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 4 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet(  ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 4 ),
    setSaturationFlags = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet(  ),
    PETstat = cms.PSet(  ),
    digistat = cms.PSet(  ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet(  ),
    correctForPhaseContainment = cms.bool( True ),
    correctForTimeslew = cms.bool( True ),
    setNoiseFlags = cms.bool( False ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HBHE" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 4 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet( 
      ignorelowest = cms.bool( True ),
      win_offset = cms.double( 0.0 ),
      ignorehighest = cms.bool( False ),
      win_gain = cms.double( 1.0 ),
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 )
    ),
    pulseShapeParameters = cms.PSet(  ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    hscpParameters = cms.PSet( 
      slopeMax = cms.double( -0.6 ),
      r1Max = cms.double( 1.0 ),
      r1Min = cms.double( 0.15 ),
      TimingEnergyThreshold = cms.double( 30.0 ),
      slopeMin = cms.double( -1.5 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      fracLeaderMin = cms.double( 0.4 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMax = cms.double( 0.7 )
    )
)
process.hltHfreco = cms.EDProducer( "HcalHitReconstructor",
    digiTimeFromDB = cms.bool( True ),
    S9S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 24 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      isS8S1 = cms.bool( False ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 2 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet( 
      hflongEthresh = cms.double( 40.0 ),
      hflongMinWindowTime = cms.vdouble( -10.0 ),
      hfshortEthresh = cms.double( 40.0 ),
      hflongMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMinWindowTime = cms.vdouble( -12.0 )
    ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 1 ),
    setSaturationFlags = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel2 = cms.int32( 4 ),
      hfTimingTrustLevel1 = cms.int32( 1 )
    ),
    PETstat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 0 ),
      short_R = cms.vdouble( 0.8 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R_29 = cms.vdouble( 0.8 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_R = cms.vdouble( 0.98 ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    digistat = cms.PSet( 
      HFdigiflagFirstSample = cms.int32( 1 ),
      HFdigiflagMinEthreshold = cms.double( 40.0 ),
      HFdigiflagSamplesToAdd = cms.int32( 3 ),
      HFdigiflagExpectedPeak = cms.int32( 2 ),
      HFdigiflagCoef = cms.vdouble( 0.93, -0.012667, -0.38275 )
    ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      flagsToSkip = cms.int32( 16 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      isS8S1 = cms.bool( True ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    correctForPhaseContainment = cms.bool( False ),
    correctForTimeslew = cms.bool( False ),
    setNoiseFlags = cms.bool( True ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HF" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 2 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet(  ),
    pulseShapeParameters = cms.PSet(  ),
    flagParameters = cms.PSet(  ),
    hscpParameters = cms.PSet(  )
)
process.hltL1SeededPhotonHcalForHE = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    eMinHE = cms.double( 0.8 ),
    hbheRecHitProducer = cms.InputTag( "hltHbhereco" ),
    effectiveAreaBarrel = cms.double( 0.105 ),
    outerCone = cms.double( 0.14 ),
    eMinHB = cms.double( 0.7 ),
    innerCone = cms.double( 0.0 ),
    etMinHE = cms.double( -1.0 ),
    etMinHB = cms.double( -1.0 ),
    rhoProducer = cms.InputTag( 'hltKT6CaloJets','rho' ),
    depth = cms.int32( -1 ),
    doRhoCorrection = cms.bool( False ),
    effectiveAreaEndcap = cms.double( 0.17 ),
    recoEcalCandidateProducer = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    rhoMax = cms.double( 9.9999999E7 ),
    rhoScale = cms.double( 1.0 ),
    doEtSum = cms.bool( False )
)
process.hltEG22CaloIdLCaloIsoVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltEG22CaloIdLCaloIsoVLEcalIsoFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltL1SeededPhotonHcalIso = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    eMinHE = cms.double( 0.8 ),
    hbheRecHitProducer = cms.InputTag( "hltHbhereco" ),
    effectiveAreaBarrel = cms.double( 0.105 ),
    outerCone = cms.double( 0.29 ),
    eMinHB = cms.double( 0.7 ),
    innerCone = cms.double( 0.16 ),
    etMinHE = cms.double( -1.0 ),
    etMinHB = cms.double( -1.0 ),
    rhoProducer = cms.InputTag( 'hltKT6CaloJets','rho' ),
    depth = cms.int32( -1 ),
    doRhoCorrection = cms.bool( False ),
    effectiveAreaEndcap = cms.double( 0.17 ),
    recoEcalCandidateProducer = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    rhoMax = cms.double( 9.9999999E7 ),
    rhoScale = cms.double( 1.0 ),
    doEtSum = cms.bool( True )
)
process.hltEG22CaloIdLCaloIsoVLHcalIsoFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.2 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.2 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalIso" ),
    candTag = cms.InputTag( "hltEG22CaloIdLCaloIsoVLHEFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltL1SeededStartUpElectronPixelSeeds = cms.EDProducer( "ElectronSeedProducer",
    endcapSuperClusters = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Seeded" ),
    SeedConfiguration = cms.PSet( 
      searchInTIDTEC = cms.bool( True ),
      HighPtThreshold = cms.double( 35.0 ),
      r2MinF = cms.double( -0.15 ),
      OrderedHitsFactoryPSet = cms.PSet( 
        maxElement = cms.uint32( 0 ),
        ComponentName = cms.string( "StandardHitPairGenerator" ),
        SeedingLayers = cms.string( "hltESPMixedLayerPairs" ),
        useOnDemandTracker = cms.untracked.int32( 0 )
      ),
      DeltaPhi1Low = cms.double( 0.23 ),
      DeltaPhi1High = cms.double( 0.08 ),
      ePhiMin1 = cms.double( -0.08 ),
      PhiMin2 = cms.double( -0.004 ),
      LowPtThreshold = cms.double( 3.0 ),
      RegionPSet = cms.PSet( 
        deltaPhiRegion = cms.double( 0.4 ),
        originHalfLength = cms.double( 15.0 ),
        useZInVertex = cms.bool( True ),
        deltaEtaRegion = cms.double( 0.1 ),
        ptMin = cms.double( 1.5 ),
        originRadius = cms.double( 0.2 ),
        VertexProducer = cms.InputTag( "dummyVertices" )
      ),
      maxHOverE = cms.double( 999999.0 ),
      dynamicPhiRoad = cms.bool( False ),
      ePhiMax1 = cms.double( 0.04 ),
      DeltaPhi2 = cms.double( 0.004 ),
      measurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
      SizeWindowENeg = cms.double( 0.675 ),
      nSigmasDeltaZ1 = cms.double( 5.0 ),
      rMaxI = cms.double( 0.2 ),
      PhiMax2 = cms.double( 0.004 ),
      preFilteredSeeds = cms.bool( True ),
      r2MaxF = cms.double( 0.15 ),
      pPhiMin1 = cms.double( -0.04 ),
      initialSeeds = cms.InputTag( "noSeedsHere" ),
      pPhiMax1 = cms.double( 0.08 ),
      hbheModule = cms.string( "hbhereco" ),
      SCEtCut = cms.double( 3.0 ),
      z2MaxB = cms.double( 0.09 ),
      fromTrackerSeeds = cms.bool( True ),
      hcalRecHits = cms.InputTag( "hltHbhereco" ),
      z2MinB = cms.double( -0.09 ),
      hbheInstance = cms.string( "" ),
      rMinI = cms.double( -0.2 ),
      hOverEConeSize = cms.double( 0.0 ),
      hOverEHBMinE = cms.double( 999999.0 ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      applyHOverECut = cms.bool( False ),
      hOverEHFMinE = cms.double( 999999.0 )
    ),
    barrelSuperClusters = cms.InputTag( "hltCorrectedHybridSuperClustersL1Seeded" )
)
process.hltEle22CaloIdLCaloIsoVLPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( True ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltEG22CaloIdLCaloIsoVLHcalIsoFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltL1sL1BeamGasHfBptxPlusPostQuiet = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BeamGas_Hf_BptxPlusPostQuiet" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreBeamGasHFBeam1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHFAsymmetryFilterTight = cms.EDFilter( "HLTHFAsymmetryFilter",
    ECut_HF = cms.double( 5.0 ),
    SS_Asym_min = cms.double( 0.45 ),
    HFHitCollection = cms.InputTag( "hltHfreco" ),
    OS_Asym_max = cms.double( -1.0 )
)
process.hltL1sL1BeamGasHfBptxMinusPostQuiet = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BeamGas_Hf_BptxMinusPostQuiet" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreBeamGasHFBeam2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1BeamHalo = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BeamHalo" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreBeamHalo = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPixelActivityFilterForHalo = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 10 ),
    saveTags = cms.bool( False ),
    inputTag = cms.InputTag( "hltSiPixelClusters" ),
    minClusters = cms.uint32( 0 )
)
process.hltTrackerHaloFilter = cms.EDFilter( "HLTTrackerHaloFilter",
    saveTags = cms.bool( False ),
    MaxAccus = cms.int32( 4 ),
    MaxClustersTEC = cms.int32( 60 ),
    MaxClustersTECm = cms.int32( 50 ),
    SignalAccumulation = cms.int32( 5 ),
    inputTag = cms.InputTag( "hltSiStripClusters" ),
    MaxClustersTECp = cms.int32( 50 ),
    FastProcessing = cms.int32( 1 )
)
process.hltPAL1EventNumberUTCA = cms.EDFilter( "HLTL1NumberFilter",
    invert = cms.bool( False ),
    period = cms.uint32( 8192 ),
    rawInput = cms.InputTag( "rawDataCollector" )
)
process.hltPrePAHcalUTCA = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1EventNumberNZS = cms.EDFilter( "HLTL1NumberFilter",
    invert = cms.bool( False ),
    period = cms.uint32( 4096 ),
    rawInput = cms.InputTag( "rawDataCollector" )
)
process.hltL1sPAHcalPhiSym = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG_13_7 OR L1_TripleEG7 OR L1_TripleEG_12_7_5 OR L1_SingleEG5_BptxAND OR L1_SingleEG7 OR L1_SingleEG12 OR L1_SingleEG18er OR L1_SingleIsoEG18er OR L1_SingleEG20 OR L1_SingleIsoEG20er OR L1_SingleEG22 OR L1_SingleEG24 OR L1_SingleEG30 OR L1_SingleMu3 OR L1_SingleMu7 OR L1_SingleMu12 OR L1_SingleMu16 OR L1_SingleMu20 OR L1_SingleMu14er OR L1_SingleMu16er OR L1_SingleMu20er OR L1_SingleMu25er OR L1_DoubleMu0 OR L1_DoubleMu5 OR L1_DoubleMu_12_5 OR L1_DoubleMu_10_Open" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAHcalPhiSym = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sPAHcalNZS = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND OR L1_SingleEG7 OR L1_SingleEG12 OR L1_SingleEG18er OR L1_SingleEG20 OR L1_SingleEG22 OR L1_SingleEG24 OR L1_SingleEG30 OR L1_SingleJet16_BptxAND OR L1_SingleJet36 OR L1_SingleJet52 OR L1_SingleJet68 OR L1_SingleJet92 OR L1_SingleJet128 OR L1_SingleMu7 OR L1_SingleMu12 OR L1_SingleMu16 OR L1_SingleMu20 OR L1_SingleMu14er OR L1_SingleMu16er OR L1_SingleMu20er OR L1_SingleMu25er OR L1_ZeroBias" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAHcalNZS = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet20CentralNoBPTXNoHalo = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJetC20_NotBptxOR" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 1 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreGlobalRunHPDNoise = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDSTPhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
process.hltPreDTCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltDTCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780 )
)
process.hltPreEcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 )
)
process.hltPreHcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter",
    InputTag = cms.InputTag( "rawDataCollector" ),
    CalibTypes = cms.vint32( 1, 2, 3, 4, 5, 6 ),
    FilterSummary = cms.untracked.bool( False )
)
process.hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731 )
)
process.hltPreTrackerCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltLaserAlignmentEventFilter = cms.EDFilter( "LaserAlignmentEventFilter",
    FED_Filter = cms.bool( True ),
    FED_IDs = cms.vint32( 260, 261, 262, 263, 264, 265, 266, 267, 269, 270, 273, 274, 277, 278, 281, 282, 284, 285, 288, 289, 292, 293, 294, 295, 300, 301, 304, 305, 308, 309, 310, 311, 316, 317, 324, 325, 329, 330, 331, 332, 339, 340, 341, 342, 349, 350, 351, 352, 164, 165, 172, 173, 177, 178, 179, 180, 187, 188, 189, 190, 197, 198, 199, 200, 204, 205, 208, 209, 212, 213, 214, 215, 220, 221, 224, 225, 228, 229, 230, 231, 236, 237, 238, 239, 240, 241, 242, 243, 245, 246, 249, 250, 253, 254, 257, 258, 478, 476, 477, 482, 484, 480, 481, 474, 459, 460, 461, 463, 485, 487, 488, 489, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 288, 289, 292, 293, 300, 301, 304, 305, 310, 311, 316, 317, 329, 330, 339, 340, 341, 342, 349, 350, 164, 165, 177, 178, 179, 180, 189, 190, 197, 198, 204, 205, 212, 213, 220, 221, 224, 225, 230, 231 ),
    SINGLE_CHANNEL_THRESH = cms.uint32( 11 ),
    FedInputTag = cms.InputTag( "rawDataCollector" ),
    DET_ID_Filter = cms.bool( True ),
    CHANNEL_COUNT_THRESH = cms.uint32( 8 ),
    SIGNAL_Filter = cms.bool( True ),
    SIGNAL_IDs = cms.vint32( 470389128, 470389384, 470389640, 470389896, 470390152, 470390408, 470390664, 470390920, 470389192, 470389448, 470389704, 470389960, 470390216, 470390472, 470390728, 470390984, 470126984, 470127240, 470127496, 470127752, 470128008, 470128264, 470128520, 470128776, 470127048, 470127304, 470127560, 470127816, 470128072, 470128328, 470128584, 470128840, 436232506, 436232826, 436233146, 436233466, 369174604, 369174812, 369175068, 369175292, 470307468, 470307716, 470308236, 470308748, 470308996, 470045316, 470045580, 470046084, 470046596, 470046860 ),
    DET_IDs = ( cms.vint32( 470307208, 470307272, 470307464, 470307528, 470307720, 470307784, 470307976, 470308040, 470308232, 470308296, 470308488, 470308552, 470308744, 470308808, 470309000, 470309064, 470323592, 470323656, 470323848, 470323912, 470324104, 470324168, 470324360, 470324424, 470324616, 470324680, 470324872, 470324936, 470325128, 470325192, 470325384, 470325448, 470339976, 470340040, 470340232, 470340296, 470340488, 470340552, 470340744, 470340808, 470341000, 470341064, 470341256, 470341320, 470341512, 470341576, 470341768, 470341832, 470356360, 470356424, 470356616, 470356680, 470356872, 470356936, 470357128, 470357192, 470357384, 470357448, 470357640, 470357704, 470357896, 470357960, 470358152, 470358216, 470372744, 470372808, 470373000, 470373064, 470373256, 470373320, 470373512, 470373576, 470373768, 470373832, 470374024, 470374088, 470374280, 470374344, 470374536, 470374600, 470389128, 470389192, 470389384, 470389448, 470389640, 470389704, 470389896, 470389960, 470390152, 470390216, 470390408, 470390472, 470390664, 470390728, 470390920, 470390984, 470405512, 470405576, 470405768, 470405832, 470406024, 470406088, 470406280, 470406344, 470406536, 470406600, 470406792, 470406856, 470407048, 470407112, 470407304, 470407368, 470421896, 470421960, 470422152, 470422216, 470422408, 470422472, 470422664, 470422728, 470422920, 470422984, 470423176, 470423240, 470423432, 470423496, 470423688, 470423752, 470438280, 470438344, 470438536, 470438600, 470438792, 470438856, 470439048, 470439112, 470439304, 470439368, 470439560, 470439624, 470439816, 470439880, 470440072, 470440136, 470045064, 470061448, 470077832, 470094216, 470110600, 470126984, 470143368, 470159752, 470176136, 470045320, 470061704, 470078088, 470094472, 470110856, 470127240, 470143624, 470160008, 470176392, 470045576, 470061960, 470078344, 470094728, 470111112, 470127496, 470143880, 470160264, 470176648, 470045832, 470062216, 470078600, 470094984, 470111368, 470127752, 470144136, 470160520, 470176904, 470046088, 470062472, 470078856, 470095240, 470111624, 470128008, 470144392, 470160776, 470177160, 470046344, 470062728, 470079112, 470095496, 470111880, 470128264, 470144648, 470161032, 470177416, 470046600, 470062984, 470079368, 470095752, 470112136, 470128520, 470144904, 470161288, 470177672, 470046856, 470063240, 470079624, 470096008, 470112392, 470128776, 470145160, 470161544, 470177928, 470045128, 470061512, 470077896, 470094280, 470110664, 470127048, 470143432, 470159816, 470176200, 470045384, 470061768, 470078152, 470094536, 470110920, 470127304, 470143688, 470160072, 470176456, 470045640, 470062024, 470078408, 470094792, 470111176, 470127560, 470143944, 470160328, 470176712, 470045896, 470062280, 470078664, 470095048, 470111432, 470127816, 470144200, 470160584, 470176968, 470046152, 470062536, 470078920)+cms.vint32( 470095304, 470111688, 470128072, 470144456, 470160840, 470177224, 470046408, 470062792, 470079176, 470095560, 470111944, 470128328, 470144712, 470161096, 470177480, 470046664, 470063048, 470079432, 470095816, 470112200, 470128584, 470144968, 470161352, 470177736, 470046920, 470063304, 470079688, 470096072, 470112456, 470128840, 470145224, 470161608, 470177992, 436232314, 436232306, 436232298, 436228198, 436228206, 436228214, 436232506, 436232498, 436232490, 436228390, 436228398, 436228406, 436232634, 436232626, 436232618, 436228518, 436228526, 436228534, 436232826, 436232818, 436232810, 436228710, 436228718, 436228726, 436233018, 436233010, 436233002, 436228902, 436228910, 436228918, 436233146, 436233138, 436233130, 436229030, 436229038, 436229046, 436233338, 436233330, 436233322, 436229222, 436229230, 436229238, 436233466, 436233458, 436233450, 436229350, 436229358, 436229366, 369174604, 369174600, 369174596, 369170500, 369170504, 369170508, 369174732, 369174728, 369174724, 369170628, 369170632, 369170636, 369174812, 369174808, 369174804, 369170708, 369170712, 369170716, 369174940, 369174936, 369174932, 369170836, 369170840, 369170844, 369175068, 369175064, 369175060, 369170964, 369170968, 369170972, 369175164, 369175160, 369175156, 369171060, 369171064, 369171068, 369175292, 369175288, 369175284, 369171188, 369171192, 369171196, 369175372, 369175368, 369175364, 369171268, 369171272, 369171276, 470307468, 470323852, 470340236, 470356620, 470373004, 470307716, 470324100, 470340484, 470356868, 470373252, 470308236, 470324620, 470341004, 470357388, 470373772, 470308748, 470325132, 470341516, 470357900, 470374284, 470308996, 470325380, 470341764, 470358148, 470374532, 470045316, 470061700, 470078084, 470094468, 470110852, 470045580, 470061964, 470078348, 470094732, 470111116, 470046084, 470062468, 470078852, 470095236, 470111620, 470046596, 470062980, 470079364, 470095748, 470112132, 470046860, 470063244, 470079628, 470096012, 470112396) )
)
process.hltTrackerCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 260, 261, 262, 263, 264, 265, 266, 267, 269, 270, 273, 274, 277, 278, 281, 282, 284, 285, 288, 289, 292, 293, 294, 295, 300, 301, 304, 305, 308, 309, 310, 311, 316, 317, 324, 325, 329, 330, 331, 332, 339, 340, 341, 342, 349, 350, 351, 352, 164, 165, 172, 173, 177, 178, 179, 180, 187, 188, 189, 190, 197, 198, 199, 200, 204, 205, 208, 209, 212, 213, 214, 215, 220, 221, 224, 225, 228, 229, 230, 231, 236, 237, 238, 239, 240, 241, 242, 243, 245, 246, 249, 250, 253, 254, 257, 258, 478, 476, 477, 482, 484, 480, 481, 474, 459, 460, 461, 463, 485, 487, 488, 489, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 288, 289, 292, 293, 300, 301, 304, 305, 310, 311, 316, 317, 329, 330, 339, 340, 341, 342, 349, 350, 164, 165, 177, 178, 179, 180, 189, 190, 197, 198, 204, 205, 212, 213, 220, 221, 224, 225, 230, 231 )
)
process.hltBPTXAntiCoincidence = cms.EDFilter( "HLTLevel1Activity",
    technicalBits = cms.uint64( 0x8 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( True ),
    physicsLoBits = cms.uint64( 0x0 ),
    physicsHiBits = cms.uint64( 0x0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    bunchCrossings = cms.vint32( 0, 1, -1 )
)
process.hltL1sL1SingleMuOpen = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreL1SingleMuOpenAntiBPTX = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1MuOpenL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMuOpen" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sTrackerCosmics = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "25" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPreL1TrackerCosmics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTrackerCosmicsPattern = cms.EDFilter( "HLTLevel1Pattern",
    ignoreL1Mask = cms.bool( False ),
    triggerPattern = cms.vint32( 1, 1, 1, 0, 0 ),
    triggerBit = cms.string( "L1Tech_RPC_TTU_pointing_Cosmics.v0" ),
    invert = cms.bool( False ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 ),
    bunchCrossings = cms.vint32( -2, -1, 0, 1, 2 )
)
process.hltL1sAlCaPAEcalPi0Eta = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND OR L1_SingleEG7 OR L1_SingleEG12 OR L1_SingleEG20 OR L1_SingleEG22 OR L1_SingleEG24 OR L1_SingleEG30 OR L1_DoubleEG_13_7 OR L1_TripleEG7 OR L1_TripleEG_12_7_5 OR L1_DoubleEG5 OR L1_TripleJet_64_44_24_VBF OR L1_TripleJet_64_48_28_VBF OR L1_TripleJetC_52_28_28 OR L1_QuadJetC32 OR L1_QuadJetC36 OR L1_QuadJetC40  OR L1_DoubleEG6_HTT100 OR L1_DoubleEG6_HTT125 OR L1_EG8_DoubleJetC20 OR L1_Mu12_EG7 OR L1_MuOpen_EG12 OR L1_DoubleMu3p5_EG5 OR L1_DoubleMu5_EG5 OR L1_Mu12_EG7 OR L1_Mu5_DoubleEG5 OR L1_Mu5_DoubleEG6 OR L1_MuOpen_EG5" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreAlCaPAEcalPi0EBonly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalRegionalPi0EtaFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet( 
    ),
    sourceTag_es = cms.InputTag( "hltESRawToRecHitFacility" ),
    doES = cms.bool( True ),
    type = cms.string( "egamma" ),
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    EmJobPSet = cms.VPSet( 
      cms.PSet(  regionEtaMargin = cms.double( 0.25 ),
        regionPhiMargin = cms.double( 0.4 ),
        Ptmin = cms.double( 2.0 ),
        Source = cms.InputTag( 'hltL1extraParticles','Isolated' )
      ),
      cms.PSet(  regionEtaMargin = cms.double( 0.25 ),
        regionPhiMargin = cms.double( 0.4 ),
        Ptmin = cms.double( 2.0 ),
        Source = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
      )
    ),
    CandJobPSet = cms.VPSet( 
    ),
    MuonJobPSet = cms.PSet(  ),
    esInstance = cms.untracked.string( "es" ),
    MuJobPSet = cms.PSet(  )
)
process.hltESRegionalPi0EtaRecHit = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( False ),
    rechitCollection = cms.string( "EcalRecHitsES" ),
    EErechitCollection = cms.string( "" ),
    EBrechitCollection = cms.string( "" ),
    sourceTag = cms.InputTag( 'hltEcalRegionalPi0EtaFEDs','es' ),
    cleaningConfig = cms.PSet(  ),
    lazyGetterTag = cms.InputTag( "hltESRawToRecHitFacility" )
)
process.hltEcalRegionalPi0EtaRecHit = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( True ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    sourceTag = cms.InputTag( "hltEcalRegionalPi0EtaFEDs" ),
    cleaningConfig = cms.PSet(  ),
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" )
)
process.hltSimple3x3Clusters = cms.EDProducer( "EgammaHLTNxNClusterProducer",
    statusLevelRecHitsToUse = cms.int32( 1 ),
    barrelClusterCollection = cms.string( "Simple3x3ClustersBarrel" ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    maxNumberofClusters = cms.int32( 38 ),
    clusPhiSize = cms.int32( 3 ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    ),
    clusEtaSize = cms.int32( 3 ),
    useRecoFlag = cms.bool( False ),
    endcapHitProducer = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEE' ),
    maxNumberofSeeds = cms.int32( 250 ),
    endcapClusterCollection = cms.string( "Simple3x3ClustersEndcap" ),
    debugLevel = cms.int32( 0 ),
    barrelHitProducer = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEB' ),
    clusSeedThrEndCap = cms.double( 1.0 ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    useDBStatus = cms.bool( True ),
    clusSeedThr = cms.double( 0.5 )
)
process.hltAlCaPi0RecHitsFilterEBonly = cms.EDFilter( "HLTEcalResonanceFilter",
    barrelSelection = cms.PSet( 
      massLowPi0Cand = cms.double( 0.084 ),
      selePtGamma = cms.double( 1.3 ),
      seleMinvMaxBarrel = cms.double( 0.23 ),
      selePtPair = cms.double( 2.6 ),
      seleMinvMinBarrel = cms.double( 0.04 ),
      seleS4S9Gamma = cms.double( 0.83 ),
      seleS9S25Gamma = cms.double( 0.0 ),
      seleIso = cms.double( 0.5 ),
      seleBeltDR = cms.double( 0.2 ),
      ptMinForIsolation = cms.double( 0.5 ),
      store5x5RecHitEB = cms.bool( False ),
      seleBeltDeta = cms.double( 0.05 ),
      removePi0CandidatesForEta = cms.bool( False ),
      barrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
      massHighPi0Cand = cms.double( 0.156 )
    ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEE' ),
    doSelBarrel = cms.bool( True ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    preshRecHitProducer = cms.InputTag( 'hltESRegionalPi0EtaRecHit','EcalRecHitsES' ),
    doSelEndcap = cms.bool( False ),
    storeRecHitES = cms.bool( True ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    barrelHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEB' ),
    useRecoFlag = cms.bool( False ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    debugLevel = cms.int32( 0 ),
    endcapSelection = cms.PSet( 
      selePtGammaEndCap_region1 = cms.double( 0.6 ),
      region2_EndCap = cms.double( 2.5 ),
      selePtGammaEndCap_region2 = cms.double( 0.6 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      region1_EndCap = cms.double( 2.0 ),
      selePtGammaEndCap_region3 = cms.double( 0.5 ),
      selePtPairMaxEndCap_region3 = cms.double( 2.5 ),
      seleMinvMinEndCap = cms.double( 0.05 ),
      seleS4S9GammaEndCap = cms.double( 0.9 ),
      seleS9S25GammaEndCap = cms.double( 0.0 ),
      selePtPairEndCap_region1 = cms.double( 2.5 ),
      seleBeltDREndCap = cms.double( 0.2 ),
      selePtPairEndCap_region3 = cms.double( 99.0 ),
      selePtPairEndCap_region2 = cms.double( 1.5 ),
      seleIsoEndCap = cms.double( 0.5 ),
      seleMinvMaxEndCap = cms.double( 0.3 ),
      endcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),
      seleBeltDetaEndCap = cms.double( 0.05 ),
      store5x5RecHitEE = cms.bool( False )
    ),
    preshowerSelection = cms.PSet( 
      preshCalibGamma = cms.double( 0.024 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 ),
      ESCollection = cms.string( "pi0EcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshClusterEnergyCut = cms.double( 0.0 ),
      preshSeededNstrip = cms.int32( 15 )
    ),
    useDBStatus = cms.bool( True )
)
process.hltAlCaPi0EBUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    doEnergyScale = cms.bool( True ),
    doLaserCorrectionsInverse = cms.bool( True ),
    EERecHitCollection = cms.InputTag( '','EcalRecHitsEE' ),
    doEnergyScaleInverse = cms.bool( True ),
    EBRecHitCollection = cms.InputTag( 'hltAlCaPi0RecHitsFilterEBonly','pi0EcalRecHitsEB' ),
    doIntercalibInverse = cms.bool( True ),
    doLaserCorrections = cms.bool( True ),
    EBRecalibRecHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    doIntercalib = cms.bool( True ),
    EERecalibRecHitCollection = cms.string( "pi0EcalRecHitsEE" )
)
process.hltPreAlCaPAEcalPi0EEonly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltAlCaPi0RecHitsFilterEEonly = cms.EDFilter( "HLTEcalResonanceFilter",
    barrelSelection = cms.PSet( 
      massLowPi0Cand = cms.double( 0.084 ),
      selePtGamma = cms.double( 1.3 ),
      seleMinvMaxBarrel = cms.double( 0.23 ),
      selePtPair = cms.double( 2.6 ),
      seleMinvMinBarrel = cms.double( 0.04 ),
      seleS4S9Gamma = cms.double( 0.83 ),
      seleS9S25Gamma = cms.double( 0.0 ),
      seleIso = cms.double( 0.5 ),
      seleBeltDR = cms.double( 0.2 ),
      ptMinForIsolation = cms.double( 0.5 ),
      store5x5RecHitEB = cms.bool( False ),
      seleBeltDeta = cms.double( 0.05 ),
      removePi0CandidatesForEta = cms.bool( False ),
      barrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
      massHighPi0Cand = cms.double( 0.156 )
    ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEE' ),
    doSelBarrel = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    preshRecHitProducer = cms.InputTag( 'hltESRegionalPi0EtaRecHit','EcalRecHitsES' ),
    doSelEndcap = cms.bool( True ),
    storeRecHitES = cms.bool( True ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    barrelHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEB' ),
    useRecoFlag = cms.bool( False ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    debugLevel = cms.int32( 0 ),
    endcapSelection = cms.PSet( 
      selePtGammaEndCap_region1 = cms.double( 0.6 ),
      region2_EndCap = cms.double( 2.5 ),
      selePtGammaEndCap_region2 = cms.double( 0.6 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      region1_EndCap = cms.double( 2.0 ),
      selePtGammaEndCap_region3 = cms.double( 0.5 ),
      selePtPairMaxEndCap_region3 = cms.double( 2.5 ),
      seleMinvMinEndCap = cms.double( 0.05 ),
      seleS4S9GammaEndCap = cms.double( 0.9 ),
      seleS9S25GammaEndCap = cms.double( 0.0 ),
      selePtPairEndCap_region1 = cms.double( 2.5 ),
      seleBeltDREndCap = cms.double( 0.2 ),
      selePtPairEndCap_region3 = cms.double( 99.0 ),
      selePtPairEndCap_region2 = cms.double( 1.5 ),
      seleIsoEndCap = cms.double( 0.5 ),
      seleMinvMaxEndCap = cms.double( 0.3 ),
      endcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),
      seleBeltDetaEndCap = cms.double( 0.05 ),
      store5x5RecHitEE = cms.bool( False )
    ),
    preshowerSelection = cms.PSet( 
      preshCalibGamma = cms.double( 0.024 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 ),
      ESCollection = cms.string( "pi0EcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshClusterEnergyCut = cms.double( 0.0 ),
      preshSeededNstrip = cms.int32( 15 )
    ),
    useDBStatus = cms.bool( True )
)
process.hltAlCaPi0EEUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    doEnergyScale = cms.bool( True ),
    doLaserCorrectionsInverse = cms.bool( True ),
    EERecHitCollection = cms.InputTag( 'hltAlCaPi0RecHitsFilterEEonly','pi0EcalRecHitsEE' ),
    doEnergyScaleInverse = cms.bool( True ),
    EBRecHitCollection = cms.InputTag( '','pi0EcalRecHitsEB' ),
    doIntercalibInverse = cms.bool( True ),
    doLaserCorrections = cms.bool( True ),
    EBRecalibRecHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    doIntercalib = cms.bool( True ),
    EERecalibRecHitCollection = cms.string( "pi0EcalRecHitsEE" )
)
process.hltPreAlCaPAEcalEtaEBonly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltAlCaEtaRecHitsFilterEBonly = cms.EDFilter( "HLTEcalResonanceFilter",
    barrelSelection = cms.PSet( 
      massLowPi0Cand = cms.double( 0.084 ),
      selePtGamma = cms.double( 1.2 ),
      seleMinvMaxBarrel = cms.double( 0.8 ),
      selePtPair = cms.double( 4.0 ),
      seleMinvMinBarrel = cms.double( 0.3 ),
      seleS4S9Gamma = cms.double( 0.87 ),
      seleS9S25Gamma = cms.double( 0.8 ),
      seleIso = cms.double( 0.5 ),
      seleBeltDR = cms.double( 0.3 ),
      ptMinForIsolation = cms.double( 0.5 ),
      store5x5RecHitEB = cms.bool( True ),
      seleBeltDeta = cms.double( 0.1 ),
      removePi0CandidatesForEta = cms.bool( True ),
      barrelHitCollection = cms.string( "etaEcalRecHitsEB" ),
      massHighPi0Cand = cms.double( 0.156 )
    ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEE' ),
    doSelBarrel = cms.bool( True ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    preshRecHitProducer = cms.InputTag( 'hltESRegionalPi0EtaRecHit','EcalRecHitsES' ),
    doSelEndcap = cms.bool( False ),
    storeRecHitES = cms.bool( True ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    barrelHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEB' ),
    useRecoFlag = cms.bool( False ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    debugLevel = cms.int32( 0 ),
    endcapSelection = cms.PSet( 
      selePtGammaEndCap_region1 = cms.double( 1.0 ),
      region2_EndCap = cms.double( 2.5 ),
      selePtGammaEndCap_region2 = cms.double( 1.0 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      region1_EndCap = cms.double( 2.0 ),
      selePtGammaEndCap_region3 = cms.double( 0.7 ),
      selePtPairMaxEndCap_region3 = cms.double( 9999.0 ),
      seleMinvMinEndCap = cms.double( 0.2 ),
      seleS4S9GammaEndCap = cms.double( 0.9 ),
      seleS9S25GammaEndCap = cms.double( 0.85 ),
      selePtPairEndCap_region1 = cms.double( 3.0 ),
      seleBeltDREndCap = cms.double( 0.3 ),
      selePtPairEndCap_region3 = cms.double( 3.0 ),
      selePtPairEndCap_region2 = cms.double( 3.0 ),
      seleIsoEndCap = cms.double( 0.5 ),
      seleMinvMaxEndCap = cms.double( 0.9 ),
      endcapHitCollection = cms.string( "etaEcalRecHitsEE" ),
      seleBeltDetaEndCap = cms.double( 0.1 ),
      store5x5RecHitEE = cms.bool( True )
    ),
    preshowerSelection = cms.PSet( 
      preshCalibGamma = cms.double( 0.024 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 ),
      ESCollection = cms.string( "etaEcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshClusterEnergyCut = cms.double( 0.0 ),
      preshSeededNstrip = cms.int32( 15 )
    ),
    useDBStatus = cms.bool( True )
)
process.hltAlCaEtaEBUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    doEnergyScale = cms.bool( True ),
    doLaserCorrectionsInverse = cms.bool( True ),
    EERecHitCollection = cms.InputTag( '','etaEcalRecHitsEE' ),
    doEnergyScaleInverse = cms.bool( True ),
    EBRecHitCollection = cms.InputTag( 'hltAlCaEtaRecHitsFilterEBonly','etaEcalRecHitsEB' ),
    doIntercalibInverse = cms.bool( True ),
    doLaserCorrections = cms.bool( True ),
    EBRecalibRecHitCollection = cms.string( "etaEcalRecHitsEB" ),
    doIntercalib = cms.bool( True ),
    EERecalibRecHitCollection = cms.string( "etaEcalRecHitsEE" )
)
process.hltPreAlCaPAEcalEtaEEonly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltAlCaEtaRecHitsFilterEEonly = cms.EDFilter( "HLTEcalResonanceFilter",
    barrelSelection = cms.PSet( 
      massLowPi0Cand = cms.double( 0.084 ),
      selePtGamma = cms.double( 1.2 ),
      seleMinvMaxBarrel = cms.double( 0.8 ),
      selePtPair = cms.double( 4.0 ),
      seleMinvMinBarrel = cms.double( 0.3 ),
      seleS4S9Gamma = cms.double( 0.87 ),
      seleS9S25Gamma = cms.double( 0.8 ),
      seleIso = cms.double( 0.5 ),
      seleBeltDR = cms.double( 0.3 ),
      ptMinForIsolation = cms.double( 0.5 ),
      store5x5RecHitEB = cms.bool( True ),
      seleBeltDeta = cms.double( 0.1 ),
      removePi0CandidatesForEta = cms.bool( True ),
      barrelHitCollection = cms.string( "etaEcalRecHitsEB" ),
      massHighPi0Cand = cms.double( 0.156 )
    ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEE' ),
    doSelBarrel = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    preshRecHitProducer = cms.InputTag( 'hltESRegionalPi0EtaRecHit','EcalRecHitsES' ),
    doSelEndcap = cms.bool( True ),
    storeRecHitES = cms.bool( True ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    barrelHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEB' ),
    useRecoFlag = cms.bool( False ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    debugLevel = cms.int32( 0 ),
    endcapSelection = cms.PSet( 
      selePtGammaEndCap_region1 = cms.double( 1.0 ),
      region2_EndCap = cms.double( 2.5 ),
      selePtGammaEndCap_region2 = cms.double( 1.0 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      region1_EndCap = cms.double( 2.0 ),
      selePtGammaEndCap_region3 = cms.double( 0.7 ),
      selePtPairMaxEndCap_region3 = cms.double( 9999.0 ),
      seleMinvMinEndCap = cms.double( 0.2 ),
      seleS4S9GammaEndCap = cms.double( 0.9 ),
      seleS9S25GammaEndCap = cms.double( 0.85 ),
      selePtPairEndCap_region1 = cms.double( 3.0 ),
      seleBeltDREndCap = cms.double( 0.3 ),
      selePtPairEndCap_region3 = cms.double( 3.0 ),
      selePtPairEndCap_region2 = cms.double( 3.0 ),
      seleIsoEndCap = cms.double( 0.5 ),
      seleMinvMaxEndCap = cms.double( 0.9 ),
      endcapHitCollection = cms.string( "etaEcalRecHitsEE" ),
      seleBeltDetaEndCap = cms.double( 0.1 ),
      store5x5RecHitEE = cms.bool( True )
    ),
    preshowerSelection = cms.PSet( 
      preshCalibGamma = cms.double( 0.024 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 ),
      ESCollection = cms.string( "etaEcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshClusterEnergyCut = cms.double( 0.0 ),
      preshSeededNstrip = cms.int32( 15 )
    ),
    useDBStatus = cms.bool( True )
)
process.hltAlCaEtaEEUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    doEnergyScale = cms.bool( True ),
    doLaserCorrectionsInverse = cms.bool( True ),
    EERecHitCollection = cms.InputTag( 'hltAlCaEtaRecHitsFilterEEonly','etaEcalRecHitsEE' ),
    doEnergyScaleInverse = cms.bool( True ),
    EBRecHitCollection = cms.InputTag( '','EcalRecHitsEB' ),
    doIntercalibInverse = cms.bool( True ),
    doLaserCorrections = cms.bool( True ),
    EBRecalibRecHitCollection = cms.string( "etaEcalRecHitsEB" ),
    doIntercalib = cms.bool( True ),
    EERecalibRecHitCollection = cms.string( "etaEcalRecHitsEE" )
)
process.hltPreAlCaEcalPhiSym = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltAlCaPhiSymStream = cms.EDFilter( "HLTEcalPhiSymFilter",
    statusThreshold = cms.uint32( 3 ),
    endcapHitCollection = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    eCut_endcap_high = cms.double( 999999.0 ),
    eCut_barrel = cms.double( 0.15 ),
    eCut_barrel_high = cms.double( 999999.0 ),
    eCut_endcap = cms.double( 0.75 ),
    useRecoFlag = cms.bool( False ),
    phiSymBarrelHitCollection = cms.string( "phiSymEcalRecHitsEB" ),
    barrelHitCollection = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    phiSymEndcapHitCollection = cms.string( "phiSymEcalRecHitsEE" )
)
process.hltAlCaPhiSymUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    doEnergyScale = cms.bool( True ),
    doLaserCorrectionsInverse = cms.bool( True ),
    EERecHitCollection = cms.InputTag( 'hltAlCaPhiSymStream','phiSymEcalRecHitsEE' ),
    doEnergyScaleInverse = cms.bool( True ),
    EBRecHitCollection = cms.InputTag( 'hltAlCaPhiSymStream','phiSymEcalRecHitsEB' ),
    doIntercalibInverse = cms.bool( True ),
    doLaserCorrections = cms.bool( True ),
    EBRecalibRecHitCollection = cms.string( "phiSymEcalRecHitsEB" ),
    doIntercalib = cms.bool( True ),
    EERecalibRecHitCollection = cms.string( "phiSymEcalRecHitsEE" )
)
process.hltL1sAlCaRPC = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7 OR L1_SingleMu14er OR L1_SingleMu16er" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreAlCaRPCMuonNoTriggers = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRPCMuonNoTriggersL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sAlCaRPC" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.6 ),
    SelectQualities = cms.vint32( 6 ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreAlCaRPCMuonNoHits = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRPCPointProducer = cms.EDProducer( "RPCPointProducer",
    cscSegments = cms.InputTag( "hltCscSegments" ),
    rangestripsRB4 = cms.untracked.double( 4.0 ),
    TrackTransformer = cms.PSet(  ),
    tracks = cms.InputTag( "NotUsed" ),
    rangestrips = cms.untracked.double( 4.0 ),
    incltrack = cms.untracked.bool( False ),
    MinCosAng = cms.untracked.double( 0.95 ),
    MaxDrb4 = cms.untracked.double( 150.0 ),
    inclcsc = cms.untracked.bool( True ),
    dt4DSegments = cms.InputTag( "hltDt4DSegments" ),
    ExtrapolatedRegion = cms.untracked.double( 0.5 ),
    incldt = cms.untracked.bool( True ),
    debug = cms.untracked.bool( False ),
    MaxD = cms.untracked.double( 80.0 )
)
process.hltRPCFilter = cms.EDFilter( "HLTRPCFilter",
    rangestrips = cms.untracked.double( 1.0 ),
    rpcDTPoints = cms.InputTag( 'hltRPCPointProducer','RPCDTExtrapolatedPoints' ),
    rpcRecHits = cms.InputTag( "hltRpcRecHits" ),
    rpcCSCPoints = cms.InputTag( 'hltRPCPointProducer','RPCCSCExtrapolatedPoints' )
)
process.hltPreAlCaRPCMuonNormalisation = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRPCMuonNormaL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sAlCaRPC" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.6 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sL1AlwaysTrue = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_AlwaysTrue" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreAlCaLumiPixels = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltFEDSelectorLumiPixels = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 )
)
process.hltPreAlCaLumiPixelsZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
process.hltPreAlCaLumiPixelsRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltBPTXCoincidence = cms.EDFilter( "HLTLevel1Activity",
    technicalBits = cms.uint64( 0x1 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( False ),
    physicsLoBits = cms.uint64( 0x1 ),
    physicsHiBits = cms.uint64( 0x40000 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    bunchCrossings = cms.vint32( 0, -1, 1 )
)
process.hltL1sL1SingleJet16BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1SingleJet16 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1SingleJet36 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePASingleForJet15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHoreco = cms.EDProducer( "HcalHitReconstructor",
    digiTimeFromDB = cms.bool( True ),
    S9S1stat = cms.PSet(  ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 4 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet(  ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 4 ),
    setSaturationFlags = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet(  ),
    PETstat = cms.PSet(  ),
    digistat = cms.PSet(  ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet(  ),
    correctForPhaseContainment = cms.bool( True ),
    correctForTimeslew = cms.bool( True ),
    setNoiseFlags = cms.bool( False ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HO" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 4 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet(  ),
    pulseShapeParameters = cms.PSet(  ),
    flagParameters = cms.PSet(  ),
    hscpParameters = cms.PSet(  )
)
process.hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( False ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "hltHoreco" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( False ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHitAll:EcalRecHitsEB','hltEcalRecHitAll:EcalRecHitsEE' ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltAntiKT5CaloJets = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( True ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "NotUsed" ),
    jetPtMin = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltCaloJetIDPassed = cms.EDProducer( "HLTCaloJetIDProducer",
    min_N90 = cms.int32( -2 ),
    min_N90hits = cms.int32( 2 ),
    min_EMF = cms.double( 1.0E-6 ),
    jetsInput = cms.InputTag( "hltAntiKT5CaloJets" ),
    JetIDParams = cms.PSet( 
      useRecHits = cms.bool( True ),
      hbheRecHitsColl = cms.InputTag( "hltHbhereco" ),
      hoRecHitsColl = cms.InputTag( "hltHoreco" ),
      hfRecHitsColl = cms.InputTag( "hltHfreco" ),
      ebRecHitsColl = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
      eeRecHitsColl = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' )
    ),
    max_EMF = cms.double( 999.0 )
)
process.hltCaloJetCorrected = cms.EDProducer( "CaloJetCorrectionProducer",
    src = cms.InputTag( "hltCaloJetIDPassed" ),
    correctors = cms.vstring( 'hltESPAK5CaloL2L3' )
)
process.hltSingleForJet15 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 15.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleForJet16 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleForJet16" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePASingleForJet25 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet25 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 25.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltPrePAJet20NoJetID = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalRegionalJetsFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet( 
      cms.PSet(  regionEtaMargin = cms.double( 1.0 ),
        regionPhiMargin = cms.double( 1.0 ),
        Ptmin = cms.double( 14.0 ),
        Source = cms.InputTag( 'hltL1extraParticles','Central' )
      ),
      cms.PSet(  regionEtaMargin = cms.double( 1.0 ),
        regionPhiMargin = cms.double( 1.0 ),
        Ptmin = cms.double( 20.0 ),
        Source = cms.InputTag( 'hltL1extraParticles','Forward' )
      ),
      cms.PSet(  regionEtaMargin = cms.double( 1.0 ),
        regionPhiMargin = cms.double( 1.0 ),
        Ptmin = cms.double( 14.0 ),
        Source = cms.InputTag( 'hltL1extraParticles','Tau' )
      )
    ),
    sourceTag_es = cms.InputTag( "NotNeededoESfalse" ),
    doES = cms.bool( False ),
    type = cms.string( "jet" ),
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    EmJobPSet = cms.VPSet( 
    ),
    CandJobPSet = cms.VPSet( 
    ),
    MuonJobPSet = cms.PSet(  ),
    esInstance = cms.untracked.string( "es" ),
    MuJobPSet = cms.PSet(  )
)
process.hltEcalRegionalJetsRecHit = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( True ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    sourceTag = cms.InputTag( "hltEcalRegionalJetsFEDs" ),
    cleaningConfig = cms.PSet( 
      e6e2thresh = cms.double( 0.04 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      tightenCrack_e1_double = cms.double( 2.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      ignoreOutOfTimeThresh = cms.double( 1.0E9 ),
      cThreshold_endcap = cms.double( 15.0 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      cThreshold_double = cms.double( 10.0 )
    ),
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" )
)
process.hltTowerMakerForJets = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( False ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "hltHoreco" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( False ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag( 'hltEcalRegionalJetsRecHit:EcalRecHitsEB','hltEcalRegionalJetsRecHit:EcalRecHitsEE' ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltAntiKT5CaloJetsRegional = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( True ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTowerMakerForJets" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "offlinePrimaryVertices" ),
    jetPtMin = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltCaloJetL1MatchedRegional = cms.EDProducer( "HLTCaloJetL1MatchProducer",
    L1CenJets = cms.InputTag( 'hltL1extraParticles','Central' ),
    DeltaR = cms.double( 0.5 ),
    L1ForJets = cms.InputTag( 'hltL1extraParticles','Forward' ),
    L1TauJets = cms.InputTag( 'hltL1extraParticles','Tau' ),
    jetsInput = cms.InputTag( "hltAntiKT5CaloJetsRegional" )
)
process.hltCaloJetCorrectedRegionalNoJetID = cms.EDProducer( "CaloJetCorrectionProducer",
    src = cms.InputTag( "hltCaloJetL1MatchedRegional" ),
    correctors = cms.vstring( 'hltESPAK5CaloL2L3' )
)
process.hltSingleJet20RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPrePAJet40NoJetID = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet40RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPrePAJet60NoJetID = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet60RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPrePAJet80NoJetID = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet80RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPrePAJet100NoJetID = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet100RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPrePAJet120NoJetID = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet120RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 120.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleJet16BptxANDinForJet20Eta2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet20Eta2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet20Eta2 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 20.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 2.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet40Eta2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet40Eta2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet40Eta2 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 40.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 2.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet60Eta2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet60Eta2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet60Eta2 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 60.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 2.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet80Eta2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet80Eta2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet80Eta2 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 80.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 2.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet100Eta2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet100Eta2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet100Eta2 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 100.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 2.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet16BptxANDinForJet20Eta3 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet20Eta3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet20Eta3 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 20.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet40Eta3 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet40Eta3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet40Eta3 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 40.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet60Eta3 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet60Eta3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet60Eta3 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 60.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet80Eta3 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet80Eta3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet80Eta3 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 80.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inForJet100Eta3 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAForJet100Eta3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleForJet100Eta3 = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 100.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 1 ),
    nPos = cms.uint32( 0 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 0 )
)
process.hltL1sL1SingleJet16BptxANDinTripleJet202020 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePATripleJet202020 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet20RegionalNoJetIDinTripleJet202020 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltSecondJet20RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltThirdJet20RegionalNoJetID = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 3 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleJet36inTripleJet402020 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePATripleJet402020 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inTripleJet602020 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePATripleJet602020 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inTripleJet802020 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePATripleJet802020 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36inTripleJet1002020 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePATripleJet1002020 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet100RegionalNoJetIDinTripleJet1002020 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleJet16BptxANDAndETM30 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_BptxAND AND L1_ETM30" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAJet40ETM30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet36AndETM30 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36 AND L1_ETM30" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAJet60ETM30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleJet60RegionalNoJetIDinJet60ETM30 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrectedRegionalNoJetID" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1DoubleMu0 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1DoubleMu0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1fL1sL1DoubleMu0L1f0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sL1DoubleMu0erHighQ = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0er_HighQ" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePADimuon0NoVertexing = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltDimuonL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0erHighQ" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltDimuonL2PreFiltered0 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltDimuonL1Filtered0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1DoubleMuOpenBptxAnd = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltL1fL1sPAL1DoubleMu0HighQL1FilteredHighQ = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenBptxAnd" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32( 7, 6, 5, 0, 0, 0, 0, 0 ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltTowerMakerForHf = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( True ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( True ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0E-99 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0E-99 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kProblematic',
      'kRecovered',
      'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( True ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag(  ),
    UseRejectedRecoveredHcalHits = cms.bool( True ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltHcalPM1Tower3GeVFilter = cms.EDFilter( "HLTHcalTowerFilter",
    saveTags = cms.bool( False ),
    MinN_HFM = cms.int32( 1 ),
    MinE_HB = cms.double( -1.0 ),
    MaxN_HB = cms.int32( 999999999 ),
    inputTag = cms.InputTag( "hltTowerMakerForHf" ),
    MaxN_HE = cms.int32( 999999999 ),
    MinE_HE = cms.double( -1.0 ),
    MinE_HF = cms.double( 3.0 ),
    MinN_HF = cms.int32( 2 ),
    MaxN_HF = cms.int32( 999999999 ),
    MinN_HFP = cms.int32( 1 )
)
process.hltPrePAL1DoubleMu0HighQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1fL1sPAL1DoubleMuOpenL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenBptxAnd" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPrePAL1DoubleMuOpen = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1fL1sPAL2DoubleMu3L1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenBptxAnd" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPrePAL2DoubleMu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL2fL1sPAL2DoubleMu3L2Filtered3 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sPAL2DoubleMu3L1Filtered0" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 3.0 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu3 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltL1fL1sMu3L1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPrePAMu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL2fL1sMu3L2Filtered3 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMu3L1Filtered0" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL3fL2sMu3L3Filtered3 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL2fL1sMu3L2Filtered3" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 3.0 )
)
process.hltL1fL1sMu7L1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu7" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPrePAMu7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL2fL1sMu7L2Filtered5 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( False ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMu7L1Filtered0" ),
    MinPt = cms.double( 5.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL3fL2sMu7L3Filtered7 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL2fL1sMu7L2Filtered5" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 7.0 )
)
process.hltL1sL1SingleMu12 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu12" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltL1fL1sMu12L1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu12" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPrePAMu12 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL2fL1sMu12L2Filtered10 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( False ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMu12L1Filtered0" ),
    MinPt = cms.double( 10.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL3fL2sMu12L3Filtered12 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL2fL1sMu12L2Filtered10" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 12.0 )
)
process.hltL1sL1Mu3JetC16WdEtaPhi2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_JetC16_WdEtaPhi2" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePABTagMuJet20Mu4 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltKT6CaloJets = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 1 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( True ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 5.0 ),
    doRhoFastjet = cms.bool( True ),
    jetAlgorithm = cms.string( "Kt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.6 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "NotUsed" ),
    jetPtMin = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltCaloJetL1FastJetCorrected = cms.EDProducer( "CaloJetCorrectionProducer",
    src = cms.InputTag( "hltCaloJetIDPassed" ),
    correctors = cms.vstring( 'hltESPAK5CaloL1L2L3' )
)
process.hltBJet20L1FastJetCentralBPH = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltCaloJetL1FastJetCorrected" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 86 )
)
process.hltBSoftMuonGetJetsFromJet20L1FastJetBPH = cms.EDProducer( "HLTCaloJetCollectionProducer",
    TriggerTypes = cms.vint32( 86 ),
    HLTObject = cms.InputTag( "hltBJet20L1FastJetCentralBPH" )
)
process.hltSelector4JetsJet20L1FastJetBPH = cms.EDFilter( "LargestEtCaloJetSelector",
    maxNumber = cms.uint32( 4 ),
    filter = cms.bool( False ),
    src = cms.InputTag( "hltBSoftMuonGetJetsFromJet20L1FastJetBPH" )
)
process.hltBSoftMuonJet20L1FastJetL25JetsBPH = cms.EDFilter( "EtMinCaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltSelector4JetsJet20L1FastJetBPH" ),
    etMin = cms.double( 20.0 )
)
process.hltBSoftMuonJet20L1FastJetL25TagInfosBPH = cms.EDProducer( "SoftLepton",
    muonSelection = cms.uint32( 0 ),
    leptons = cms.InputTag( "hltL2Muons" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptonCands = cms.InputTag( "" ),
    leptonId = cms.InputTag( "" ),
    refineJetAxis = cms.uint32( 0 ),
    jets = cms.InputTag( "hltBSoftMuonJet20L1FastJetL25JetsBPH" ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 )
)
process.hltBSoftMuonJet20L1FastJetL25BJetTagsByDRBPH = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "hltESPSoftLeptonByDistance" ),
    tagInfos = cms.VInputTag( 'hltBSoftMuonJet20L1FastJetL25TagInfosBPH' )
)
process.hltBSoftMuonJet20L1FastJetL25FilterByDRBPH = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( False ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltBSoftMuonJet20L1FastJetL25BJetTagsByDRBPH" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltBSoftMuonJet20L1FastJetL25JetsBPH" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 )
)
process.hltBSoftMuonMu4L3 = cms.EDFilter( "RecoTrackRefSelector",
    src = cms.InputTag( "hltL3Muons" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    algorithm = cms.vstring(  ),
    maxChi2 = cms.double( 10.0 ),
    tip = cms.double( 120.0 ),
    quality = cms.vstring(  ),
    minRapidity = cms.double( -2.4 ),
    lip = cms.double( 300.0 ),
    ptMin = cms.double( 4.0 ),
    maxRapidity = cms.double( 2.4 ),
    min3DHit = cms.int32( 0 ),
    minHit = cms.int32( 10 )
)
process.hltBSoftMuonJet20L1FastJetMu5SelL3TagInfosBPH = cms.EDProducer( "SoftLepton",
    muonSelection = cms.uint32( 0 ),
    leptons = cms.InputTag( "hltBSoftMuonMu4L3" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptonCands = cms.InputTag( "" ),
    leptonId = cms.InputTag( "" ),
    refineJetAxis = cms.uint32( 0 ),
    jets = cms.InputTag( "hltBSoftMuonJet20L1FastJetL25JetsBPH" ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 )
)
process.hltBSoftMuonJet20L1FastJetMu5SelL3BJetTagsByDRBPH = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "hltESPSoftLeptonByDistance" ),
    tagInfos = cms.VInputTag( 'hltBSoftMuonJet20L1FastJetMu5SelL3TagInfosBPH' )
)
process.hltBSoftMuonJet20L1FastJetMu5L3FilterByDRBPH = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltBSoftMuonJet20L1FastJetMu5SelL3BJetTagsByDRBPH" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltBSoftMuonJet20L1FastJetL25JetsBPH" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 )
)
process.hltL1sL1SingleMu3Jet16 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_Jet16" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMu3PFJet20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAMu3PFJet20L2Filtered3 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( False ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3Jet16" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPAMu3PFJet20L3Filter3 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltPAMu3PFJet20L2Filtered3" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 3.0 )
)
process.hltTowerMakerForPF = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( False ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.4 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.4 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 1.8 ),
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 1.1 ),
    HOThresholdPlus2 = cms.double( 1.1 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "hltHoreco" ),
    HF1Threshold = cms.double( 1.2 ),
    HOThresholdMinus1 = cms.double( 1.1 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.4 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( True ),
    HOThresholdMinus2 = cms.double( 1.1 ),
    HOThreshold0 = cms.double( 1.1 ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHitAll:EcalRecHitsEB','hltEcalRecHitAll:EcalRecHitsEE' ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltAntiKT5CaloJetsPF = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( -9.0 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 0 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTowerMakerForPF" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "NotUsed" ),
    jetPtMin = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltAntiKT5CaloJetsPFEt5 = cms.EDFilter( "EtMinCaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltAntiKT5CaloJetsPF" ),
    etMin = cms.double( 5.0 )
)
process.hltPixelTracks = cms.EDProducer( "PixelTrackProducer",
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      ptMin = cms.double( 0.1 ),
      tipMax = cms.double( 1.0 )
    ),
    useFilterWithES = cms.bool( False ),
    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      fixImpactParameter = cms.double( 0.0 )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 0.9 ),
        originHalfLength = cms.double( 24.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 100000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useMultScattering = cms.bool( True ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ) ),
        extraHitRZtolerance = cms.double( 0.06 ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" )
      ),
      SeedingLayers = cms.string( "hltESPPixelLayerTriplets" )
    )
)
process.hltPixelVertices = cms.EDProducer( "PixelVertexProducer",
    WtAverage = cms.bool( True ),
    Method2 = cms.bool( True ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Verbosity = cms.int32( 0 ),
    UseError = cms.bool( True ),
    TrackCollection = cms.InputTag( "hltPixelTracks" ),
    PtMin = cms.double( 1.0 ),
    NTrkMin = cms.int32( 2 ),
    ZOffset = cms.double( 5.0 ),
    Finder = cms.string( "DivisiveVertexFinder" ),
    ZSeparation = cms.double( 0.05 )
)
process.hltPFJetPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 0.3 ),
    useProtoTrackKinematics = cms.bool( False ),
    InputVertexCollection = cms.InputTag( "hltPixelVertices" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    InputCollection = cms.InputTag( "hltPixelTracks" ),
    originRadius = cms.double( 0.1 )
)
process.hltPFJetCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltPFJetPixelSeedsFromPixelTracks" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPTrajectoryBuilderIT" )
)
process.hltPFJetCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltPFJetCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter0" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltPFlowTrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.4, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.35, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPFJetCtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPixelVertices" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.4, 4.0 ),
    d0_par1 = cms.vdouble( 0.3, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltTrackRefsForJetsIter0 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltPFlowTrackSelectionHighPurity" ),
    particleType = cms.string( "pi+" )
)
process.hltAntiKT5TrackJetsIter0 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTrackRefsForJetsIter0" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPixelVertices" ),
    jetPtMin = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltTrackAndTauJetsIter0 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 1.0 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltAntiKT5TrackJetsIter0" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.7 ),
    etaMaxCaloJet = cms.double( 2.7 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltPFlowTrackSelectionHighPurity" )
)
process.hltIter1ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltPFlowTrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 9.0 ) ),
    doPixel = cms.bool( True )
)
process.hltIter1SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter1ESPMeasurementTracker" )
)
process.hltIter1PFJetPixelSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 1.0 ),
        originHalfLength = cms.double( 0.1 ),
        originRadius = cms.double( 0.05 ),
        measurementTrackerName = cms.string( "hltIter1ESPMeasurementTracker" ),
        deltaEtaRegion = cms.double( 1.0 ),
        vertexSrc = cms.InputTag( "hltPixelVertices" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltTrackAndTauJetsIter0" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 0.5 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 100000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter1ESPPixelLayerTriplets" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsTripletOnlyCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltIter1PFJetCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltIter1PFJetPixelSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter1ESPTrajectoryBuilderIT" )
)
process.hltIter1PFJetCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltIter1PFJetCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter1" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltIter1PFlowTrackSelectionHighPurityLoose = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.9, 3.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.8, 3.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltIter1PFJetCtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPixelVertices" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.9, 3.0 ),
    d0_par1 = cms.vdouble( 0.85, 3.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltIter1PFlowTrackSelectionHighPurityTight = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 5 ),
    chi2n_par = cms.double( 0.4 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 1.0, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 1.0, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltIter1PFJetCtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPixelVertices" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 1.0, 4.0 ),
    d0_par1 = cms.vdouble( 1.0, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltIter1PFlowTrackSelectionHighPurity = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltIter1PFlowTrackSelectionHighPurityLoose" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltIter1PFlowTrackSelectionHighPurityTight" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltIter1Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltPFlowTrackSelectionHighPurity" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltIter1PFlowTrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltTrackRefsForJetsIter1 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltIter1Merged" ),
    particleType = cms.string( "pi+" )
)
process.hltAntiKT5TrackJetsIter1 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTrackRefsForJetsIter1" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPixelVertices" ),
    jetPtMin = cms.double( 1.4 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltTrackAndTauJetsIter1 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 1.4 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltAntiKT5TrackJetsIter1" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.7 ),
    etaMaxCaloJet = cms.double( 2.7 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltIter1Merged" )
)
process.hltIter2ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltIter1PFlowTrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "hltIter1ClustersRefRemoval" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 16.0 ) ),
    doPixel = cms.bool( True )
)
process.hltIter2SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter2ESPMeasurementTracker" )
)
process.hltIter2PFJetPixelSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 0.8 ),
        originHalfLength = cms.double( 0.05 ),
        originRadius = cms.double( 0.025 ),
        measurementTrackerName = cms.string( "hltIter2ESPMeasurementTracker" ),
        deltaEtaRegion = cms.double( 0.8 ),
        vertexSrc = cms.InputTag( "hltPixelVertices" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltTrackAndTauJetsIter1" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 1.2 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      GeneratorPSet = cms.PSet( 
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter2ESPPixelLayerPairs" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltIter2PFJetCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltIter2PFJetPixelSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter2ESPTrajectoryBuilderIT" )
)
process.hltIter2PFJetCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltIter2PFJetCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter2" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltIter2PFlowTrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.4, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.35, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltIter2PFJetCtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPixelVertices" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.4, 4.0 ),
    d0_par1 = cms.vdouble( 0.3, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltIter2Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltIter1Merged" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltIter2PFlowTrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltTrackRefsForJetsIter2 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltIter2Merged" ),
    particleType = cms.string( "pi+" )
)
process.hltAntiKT5TrackJetsIter2 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTrackRefsForJetsIter2" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPixelVertices" ),
    jetPtMin = cms.double( 3.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltTrackAndTauJetsIter2 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 3.0 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltAntiKT5TrackJetsIter2" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.7 ),
    etaMaxCaloJet = cms.double( 2.7 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltIter2Merged" )
)
process.hltIter3ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltIter2PFlowTrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "hltIter2ClustersRefRemoval" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 16.0 ) ),
    doPixel = cms.bool( True )
)
process.hltIter3SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter3ESPMeasurementTracker" )
)
process.hltIter3PFJetMixedSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 0.5 ),
        originHalfLength = cms.double( 0.05 ),
        originRadius = cms.double( 0.05 ),
        measurementTrackerName = cms.string( "hltIter3ESPMeasurementTracker" ),
        deltaEtaRegion = cms.double( 0.5 ),
        vertexSrc = cms.InputTag( "hltPixelVertices" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltTrackAndTauJetsIter2" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 0.8 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 100000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter3ESPLayerTriplets" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsTripletOnlyCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltIter3PFJetCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltIter3PFJetMixedSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter3ESPTrajectoryBuilderIT" )
)
process.hltIter3PFJetCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltIter3PFJetCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter3" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltIter3PFlowTrackSelectionHighPurityLoose = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.9, 3.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.85, 3.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltIter3PFJetCtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPixelVertices" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.9, 3.0 ),
    d0_par1 = cms.vdouble( 0.85, 3.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltIter3PFlowTrackSelectionHighPurityTight = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 5 ),
    chi2n_par = cms.double( 0.4 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 1.0, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 1.0, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltIter3PFJetCtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPixelVertices" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 1.0, 4.0 ),
    d0_par1 = cms.vdouble( 1.0, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltIter3PFlowTrackSelectionHighPurity = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltIter3PFlowTrackSelectionHighPurityLoose" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltIter3PFlowTrackSelectionHighPurityTight" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltIter3Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltIter2Merged" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltIter3PFlowTrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltTrackRefsForJetsIter3 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltIter3Merged" ),
    particleType = cms.string( "pi+" )
)
process.hltAntiKT5TrackJetsIter3 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTrackRefsForJetsIter3" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPixelVertices" ),
    jetPtMin = cms.double( 4.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltTrackAndTauJetsIter3 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 4.0 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltAntiKT5TrackJetsIter3" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.0 ),
    etaMaxCaloJet = cms.double( 2.0 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltIter3Merged" )
)
process.hltIter4ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltIter3PFlowTrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "hltIter3ClustersRefRemoval" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 16.0 ) ),
    doPixel = cms.bool( True )
)
process.hltIter4SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter4ESPMeasurementTracker" )
)
process.hltIter4PFJetPixelLessSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 0.5 ),
        originHalfLength = cms.double( 1.0 ),
        originRadius = cms.double( 0.5 ),
        measurementTrackerName = cms.string( "hltIter4ESPMeasurementTracker" ),
        deltaEtaRegion = cms.double( 0.5 ),
        vertexSrc = cms.InputTag( "hltPixelVertices" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltTrackAndTauJetsIter3" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 0.8 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      GeneratorPSet = cms.PSet( 
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter4ESPPixelLayerPairs" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltIter4PFJetCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltIter4PFJetPixelLessSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter4ESPTrajectoryBuilderIT" )
)
process.hltIter4PFJetCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltIter4PFJetCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter4" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltIter4PFlowTrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 5 ),
    chi2n_par = cms.double( 0.25 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 1.0, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 1.0, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 0 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltIter4PFJetCtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPixelVertices" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 1.0, 4.0 ),
    d0_par1 = cms.vdouble( 1.0, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltIter4Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltIter3Merged" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltIter4PFlowTrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltPFMuonMerging = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltL3TkTracksFromL2" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltIter4Merged" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltMuonLinks = cms.EDProducer( "MuonLinksProducerForHLT",
    pMin = cms.double( 2.5 ),
    InclusiveTrackerTrackCollection = cms.InputTag( "hltPFMuonMerging" ),
    shareHitFraction = cms.double( 0.8 ),
    LinkCollection = cms.InputTag( "hltL3MuonsLinksCombination" ),
    ptMin = cms.double( 2.5 )
)
process.hltMuons = cms.EDProducer( "MuonIdProducer",
    TrackExtractorPSet = cms.PSet( 
      Diff_z = cms.double( 0.2 ),
      inputTrackCollection = cms.InputTag( "hltPFMuonMerging" ),
      BeamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
      ComponentName = cms.string( "TrackExtractor" ),
      DR_Max = cms.double( 1.0 ),
      Diff_r = cms.double( 0.1 ),
      Chi2Prob_Min = cms.double( -1.0 ),
      DR_Veto = cms.double( 0.01 ),
      NHits_Min = cms.uint32( 0 ),
      Chi2Ndof_Max = cms.double( 1.0E64 ),
      Pt_Min = cms.double( -1.0 ),
      DepositLabel = cms.untracked.string( "" ),
      BeamlineOption = cms.string( "BeamSpotFromEvent" )
    ),
    maxAbsEta = cms.double( 3.0 ),
    fillGlobalTrackRefits = cms.bool( False ),
    arbitrationCleanerOptions = cms.PSet( 
      Clustering = cms.bool( True ),
      ME1a = cms.bool( True ),
      ClusterDPhi = cms.double( 0.6 ),
      OverlapDTheta = cms.double( 0.02 ),
      Overlap = cms.bool( True ),
      OverlapDPhi = cms.double( 0.0786 ),
      ClusterDTheta = cms.double( 0.02 )
    ),
    globalTrackQualityInputTag = cms.InputTag( "glbTrackQual" ),
    addExtraSoftMuons = cms.bool( False ),
    debugWithTruthMatching = cms.bool( False ),
    CaloExtractorPSet = cms.PSet( 
      PrintTimeReport = cms.untracked.bool( False ),
      DR_Max = cms.double( 1.0 ),
      DepositInstanceLabels = cms.vstring( 'ecal',
        'hcal',
        'ho' ),
      Noise_HE = cms.double( 0.2 ),
      NoiseTow_EB = cms.double( 0.04 ),
      NoiseTow_EE = cms.double( 0.15 ),
      Threshold_H = cms.double( 0.5 ),
      ServiceParameters = cms.PSet( 
        Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False )
      ),
      Threshold_E = cms.double( 0.2 ),
      PropagatorName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      DepositLabel = cms.untracked.string( "Cal" ),
      UseRecHitsFlag = cms.bool( False ),
      TrackAssociatorParameters = cms.PSet( 
        muonMaxDistanceSigmaX = cms.double( 0.0 ),
        muonMaxDistanceSigmaY = cms.double( 0.0 ),
        CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
        dRHcal = cms.double( 1.0 ),
        dRPreshowerPreselection = cms.double( 0.2 ),
        CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForPF" ),
        useEcal = cms.bool( False ),
        dREcal = cms.double( 1.0 ),
        dREcalPreselection = cms.double( 1.0 ),
        HORecHitCollectionLabel = cms.InputTag( "hltHoreco" ),
        dRMuon = cms.double( 9999.0 ),
        propagateAllDirections = cms.bool( True ),
        muonMaxDistanceX = cms.double( 5.0 ),
        muonMaxDistanceY = cms.double( 5.0 ),
        useHO = cms.bool( False ),
        trajectoryUncertaintyTolerance = cms.double( -1.0 ),
        usePreshower = cms.bool( False ),
        DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
        EERecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
        dRHcalPreselection = cms.double( 1.0 ),
        useMuon = cms.bool( False ),
        useCalo = cms.bool( True ),
        accountForTrajectoryChangeCalo = cms.bool( False ),
        EBRecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
        dRMuonPreselection = cms.double( 0.2 ),
        truthMatch = cms.bool( False ),
        HBHERecHitCollectionLabel = cms.InputTag( "hltHbhereco" ),
        useHcal = cms.bool( False )
      ),
      Threshold_HO = cms.double( 0.5 ),
      Noise_EE = cms.double( 0.1 ),
      Noise_EB = cms.double( 0.025 ),
      DR_Veto_H = cms.double( 0.1 ),
      CenterConeOnCalIntersection = cms.bool( False ),
      ComponentName = cms.string( "CaloExtractorByAssociator" ),
      Noise_HB = cms.double( 0.2 ),
      DR_Veto_E = cms.double( 0.07 ),
      DR_Veto_HO = cms.double( 0.1 ),
      Noise_HO = cms.double( 0.2 )
    ),
    runArbitrationCleaner = cms.bool( False ),
    fillEnergy = cms.bool( True ),
    TrackerKinkFinderParameters = cms.PSet( 
      usePosition = cms.bool( False ),
      diagonalOnly = cms.bool( False )
    ),
    TimingFillerParameters = cms.PSet( 
      UseDT = cms.bool( True ),
      ErrorDT = cms.double( 6.0 ),
      EcalEnergyCut = cms.double( 0.4 ),
      ErrorEB = cms.double( 2.085 ),
      ErrorCSC = cms.double( 7.4 ),
      CSCTimingParameters = cms.PSet( 
        CSCsegments = cms.InputTag( "hltCscSegments" ),
        CSCTimeOffset = cms.double( 0.0 ),
        CSCStripTimeOffset = cms.double( 0.0 ),
        MatchParameters = cms.PSet( 
          CSCsegments = cms.InputTag( "hltCscSegments" ),
          DTsegments = cms.InputTag( "hltDt4DSegments" ),
          DTradius = cms.double( 0.01 ),
          TightMatchDT = cms.bool( False ),
          TightMatchCSC = cms.bool( True )
        ),
        debug = cms.bool( False ),
        UseStripTime = cms.bool( True ),
        CSCStripError = cms.double( 7.0 ),
        CSCWireError = cms.double( 8.6 ),
        CSCWireTimeOffset = cms.double( 0.0 ),
        ServiceParameters = cms.PSet( 
          Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
          RPCLayers = cms.bool( True )
        ),
        PruneCut = cms.double( 100.0 ),
        UseWireTime = cms.bool( True )
      ),
      DTTimingParameters = cms.PSet( 
        HitError = cms.double( 6.0 ),
        DoWireCorr = cms.bool( False ),
        MatchParameters = cms.PSet( 
          CSCsegments = cms.InputTag( "hltCscSegments" ),
          DTsegments = cms.InputTag( "hltDt4DSegments" ),
          DTradius = cms.double( 0.01 ),
          TightMatchDT = cms.bool( False ),
          TightMatchCSC = cms.bool( True )
        ),
        debug = cms.bool( False ),
        DTsegments = cms.InputTag( "hltDt4DSegments" ),
        PruneCut = cms.double( 10000.0 ),
        RequireBothProjections = cms.bool( False ),
        HitsMin = cms.int32( 5 ),
        DTTimeOffset = cms.double( 2.7 ),
        DropTheta = cms.bool( True ),
        UseSegmentT0 = cms.bool( False ),
        ServiceParameters = cms.PSet( 
          Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
          RPCLayers = cms.bool( True )
        )
      ),
      ErrorEE = cms.double( 6.95 ),
      UseCSC = cms.bool( True ),
      UseECAL = cms.bool( True )
    ),
    inputCollectionTypes = cms.vstring( 'inner tracks',
      'links',
      'outer tracks' ),
    minCaloCompatibility = cms.double( 0.6 ),
    ecalDepositName = cms.string( "ecal" ),
    minP = cms.double( 10.0 ),
    fillIsolation = cms.bool( True ),
    jetDepositName = cms.string( "jets" ),
    hoDepositName = cms.string( "ho" ),
    writeIsoDeposits = cms.bool( False ),
    maxAbsPullX = cms.double( 4.0 ),
    maxAbsPullY = cms.double( 9999.0 ),
    minPt = cms.double( 10.0 ),
    TrackAssociatorParameters = cms.PSet( 
      muonMaxDistanceSigmaX = cms.double( 0.0 ),
      muonMaxDistanceSigmaY = cms.double( 0.0 ),
      CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
      dRHcal = cms.double( 9999.0 ),
      dRPreshowerPreselection = cms.double( 0.2 ),
      CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForPF" ),
      useEcal = cms.bool( True ),
      dREcal = cms.double( 9999.0 ),
      dREcalPreselection = cms.double( 0.05 ),
      HORecHitCollectionLabel = cms.InputTag( "hltHoreco" ),
      dRMuon = cms.double( 9999.0 ),
      propagateAllDirections = cms.bool( True ),
      muonMaxDistanceX = cms.double( 5.0 ),
      muonMaxDistanceY = cms.double( 5.0 ),
      useHO = cms.bool( True ),
      trajectoryUncertaintyTolerance = cms.double( -1.0 ),
      usePreshower = cms.bool( False ),
      DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
      EERecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
      dRHcalPreselection = cms.double( 0.2 ),
      useMuon = cms.bool( True ),
      useCalo = cms.bool( False ),
      accountForTrajectoryChangeCalo = cms.bool( False ),
      EBRecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
      dRMuonPreselection = cms.double( 0.2 ),
      truthMatch = cms.bool( False ),
      HBHERecHitCollectionLabel = cms.InputTag( "hltHbhereco" ),
      useHcal = cms.bool( True )
    ),
    JetExtractorPSet = cms.PSet( 
      PrintTimeReport = cms.untracked.bool( False ),
      ExcludeMuonVeto = cms.bool( True ),
      TrackAssociatorParameters = cms.PSet( 
        muonMaxDistanceSigmaX = cms.double( 0.0 ),
        muonMaxDistanceSigmaY = cms.double( 0.0 ),
        CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
        dRHcal = cms.double( 0.5 ),
        dRPreshowerPreselection = cms.double( 0.2 ),
        CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForPF" ),
        useEcal = cms.bool( False ),
        dREcal = cms.double( 0.5 ),
        dREcalPreselection = cms.double( 0.5 ),
        HORecHitCollectionLabel = cms.InputTag( "hltHoreco" ),
        dRMuon = cms.double( 9999.0 ),
        propagateAllDirections = cms.bool( True ),
        muonMaxDistanceX = cms.double( 5.0 ),
        muonMaxDistanceY = cms.double( 5.0 ),
        useHO = cms.bool( False ),
        trajectoryUncertaintyTolerance = cms.double( -1.0 ),
        usePreshower = cms.bool( False ),
        DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
        EERecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
        dRHcalPreselection = cms.double( 0.5 ),
        useMuon = cms.bool( False ),
        useCalo = cms.bool( True ),
        accountForTrajectoryChangeCalo = cms.bool( False ),
        EBRecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
        dRMuonPreselection = cms.double( 0.2 ),
        truthMatch = cms.bool( False ),
        HBHERecHitCollectionLabel = cms.InputTag( "hltHbhereco" ),
        useHcal = cms.bool( False )
      ),
      ServiceParameters = cms.PSet( 
        Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False )
      ),
      ComponentName = cms.string( "JetExtractor" ),
      DR_Max = cms.double( 1.0 ),
      PropagatorName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      JetCollectionLabel = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
      DR_Veto = cms.double( 0.1 ),
      Threshold = cms.double( 5.0 )
    ),
    fillGlobalTrackQuality = cms.bool( False ),
    minPCaloMuon = cms.double( 1.0E9 ),
    maxAbsDy = cms.double( 9999.0 ),
    fillCaloCompatibility = cms.bool( True ),
    fillMatching = cms.bool( True ),
    MuonCaloCompatibility = cms.PSet( 
      allSiPMHO = cms.bool( False ),
      PionTemplateFileName = cms.FileInPath( "RecoMuon/MuonIdentification/data/MuID_templates_pions_lowPt_3_1_norm.root" ),
      MuonTemplateFileName = cms.FileInPath( "RecoMuon/MuonIdentification/data/MuID_templates_muons_lowPt_3_1_norm.root" ),
      delta_eta = cms.double( 0.02 ),
      delta_phi = cms.double( 0.02 )
    ),
    fillTrackerKink = cms.bool( False ),
    hcalDepositName = cms.string( "hcal" ),
    sigmaThresholdToFillCandidateP4WithGlobalFit = cms.double( 2.0 ),
    inputCollectionLabels = cms.VInputTag( 'hltPFMuonMerging','hltMuonLinks','hltL2Muons' ),
    trackDepositName = cms.string( "tracker" ),
    maxAbsDx = cms.double( 3.0 ),
    ptThresholdToFillCandidateP4WithGlobalFit = cms.double( 200.0 ),
    minNumberOfMatches = cms.int32( 1 )
)
process.hltParticleFlowRecHitECAL = cms.EDProducer( "PFRecHitProducerECAL",
    crossBarrelEndcapBorder = cms.bool( False ),
    verbose = cms.untracked.bool( False ),
    ecalRecHitsEE = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    ecalRecHitsEB = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    thresh_Cleaning_EB = cms.double( 2.0 ),
    timing_Cleaning = cms.bool( True ),
    thresh_Barrel = cms.double( 0.08 ),
    thresh_Cleaning_EE = cms.double( 1.0E9 ),
    topological_Cleaning = cms.bool( True ),
    thresh_Endcap = cms.double( 0.3 )
)
process.hltParticleFlowRecHitHCAL = cms.EDProducer( "PFRecHitProducerHCAL",
    ECAL_Compensate = cms.bool( False ),
    ECAL_Dead_Code = cms.uint32( 10 ),
    MinLongTiming_Cut = cms.double( -5.0 ),
    verbose = cms.untracked.bool( False ),
    ECAL_Compensation = cms.double( 0.5 ),
    MaxLongTiming_Cut = cms.double( 5.0 ),
    weight_HFhad = cms.double( 1.0 ),
    ApplyPulseDPG = cms.bool( True ),
    ECAL_Threshold = cms.double( 10.0 ),
    ApplyTimeDPG = cms.bool( False ),
    caloTowers = cms.InputTag( "hltTowerMakerForPF" ),
    hcalRecHitsHBHE = cms.InputTag( "hltHbhereco" ),
    LongFibre_Fraction = cms.double( 0.1 ),
    MaxShortTiming_Cut = cms.double( 5.0 ),
    HcalMaxAllowedHFLongShortSev = cms.int32( 9 ),
    thresh_Barrel = cms.double( 0.4 ),
    navigation_HF = cms.bool( True ),
    HcalMaxAllowedHFInTimeWindowSev = cms.int32( 9 ),
    HF_Calib_29 = cms.double( 1.07 ),
    LongFibre_Cut = cms.double( 120.0 ),
    EM_Depth = cms.double( 22.0 ),
    weight_HFem = cms.double( 1.0 ),
    LongShortFibre_Cut = cms.double( 30.0 ),
    MinShortTiming_Cut = cms.double( -5.0 ),
    HCAL_Calib = cms.bool( True ),
    thresh_HF = cms.double( 0.4 ),
    HcalMaxAllowedHFDigiTimeSev = cms.int32( 9 ),
    thresh_Endcap = cms.double( 0.4 ),
    HcalMaxAllowedChannelStatusSev = cms.int32( 9 ),
    hcalRecHitsHF = cms.InputTag( "hltHfreco" ),
    ShortFibre_Cut = cms.double( 60.0 ),
    ApplyLongShortDPG = cms.bool( True ),
    HF_Calib = cms.bool( True ),
    HAD_Depth = cms.double( 47.0 ),
    ShortFibre_Fraction = cms.double( 0.01 ),
    HCAL_Calib_29 = cms.double( 1.35 )
)
process.hltParticleFlowRecHitPS = cms.EDProducer( "PFRecHitProducerPS",
    ecalRecHitsES = cms.InputTag( 'hltESRecHitAll','EcalRecHitsES' ),
    thresh_Barrel = cms.double( 7.0E-6 ),
    verbose = cms.untracked.bool( False ),
    thresh_Endcap = cms.double( 7.0E-6 )
)
process.hltParticleFlowClusterECAL = cms.EDProducer( "PFClusterProducer",
    posCalcNCrystal = cms.int32( 9 ),
    verbose = cms.untracked.bool( False ),
    showerSigma = cms.double( 1.5 ),
    thresh_DoubleSpike_Barrel = cms.double( 10.0 ),
    thresh_Pt_Barrel = cms.double( 0.0 ),
    thresh_Clean_Barrel = cms.double( 4.0 ),
    PFRecHits = cms.InputTag( "hltParticleFlowRecHitECAL" ),
    cleanRBXandHPDs = cms.bool( False ),
    depthCor_B = cms.double( 7.4 ),
    depthCor_A = cms.double( 0.89 ),
    nNeighbours = cms.int32( 8 ),
    thresh_DoubleSpike_Endcap = cms.double( 1.0E9 ),
    minS4S1_Clean_Barrel = cms.vdouble( 0.04, -0.024 ),
    thresh_Pt_Seed_Barrel = cms.double( 0.0 ),
    thresh_Pt_Endcap = cms.double( 0.0 ),
    thresh_Barrel = cms.double( 0.08 ),
    thresh_Clean_Endcap = cms.double( 15.0 ),
    thresh_Seed_Barrel = cms.double( 0.23 ),
    depthCor_Mode = cms.int32( 1 ),
    depthCor_B_preshower = cms.double( 4.0 ),
    useCornerCells = cms.bool( True ),
    depthCor_A_preshower = cms.double( 0.89 ),
    thresh_Endcap = cms.double( 0.3 ),
    thresh_Pt_Seed_Endcap = cms.double( 0.15 ),
    minS4S1_Clean_Endcap = cms.vdouble( 0.02, -0.0125 ),
    thresh_Seed_Endcap = cms.double( 0.6 ),
    minS6S2_DoubleSpike_Endcap = cms.double( -1.0 ),
    minS6S2_DoubleSpike_Barrel = cms.double( 0.04 )
)
process.hltParticleFlowClusterHCAL = cms.EDProducer( "PFClusterProducer",
    posCalcNCrystal = cms.int32( 5 ),
    verbose = cms.untracked.bool( False ),
    showerSigma = cms.double( 10.0 ),
    thresh_DoubleSpike_Barrel = cms.double( 1.0E9 ),
    thresh_Pt_Barrel = cms.double( 0.0 ),
    thresh_Clean_Barrel = cms.double( 100000.0 ),
    PFRecHits = cms.InputTag( "hltParticleFlowRecHitHCAL" ),
    cleanRBXandHPDs = cms.bool( True ),
    depthCor_B = cms.double( 7.4 ),
    depthCor_A = cms.double( 0.89 ),
    nNeighbours = cms.int32( 4 ),
    thresh_DoubleSpike_Endcap = cms.double( 1.0E9 ),
    minS4S1_Clean_Barrel = cms.vdouble( 0.032, -0.045 ),
    thresh_Pt_Seed_Barrel = cms.double( 0.0 ),
    thresh_Pt_Endcap = cms.double( 0.0 ),
    thresh_Barrel = cms.double( 0.8 ),
    thresh_Clean_Endcap = cms.double( 100000.0 ),
    thresh_Seed_Barrel = cms.double( 0.8 ),
    depthCor_Mode = cms.int32( 0 ),
    depthCor_B_preshower = cms.double( 4.0 ),
    useCornerCells = cms.bool( True ),
    depthCor_A_preshower = cms.double( 0.89 ),
    thresh_Endcap = cms.double( 0.8 ),
    thresh_Pt_Seed_Endcap = cms.double( 0.0 ),
    minS4S1_Clean_Endcap = cms.vdouble( 0.032, -0.045 ),
    thresh_Seed_Endcap = cms.double( 1.1 ),
    minS6S2_DoubleSpike_Endcap = cms.double( -1.0 ),
    minS6S2_DoubleSpike_Barrel = cms.double( -1.0 )
)
process.hltParticleFlowClusterHFEM = cms.EDProducer( "PFClusterProducer",
    posCalcNCrystal = cms.int32( 5 ),
    verbose = cms.untracked.bool( False ),
    showerSigma = cms.double( 10.0 ),
    thresh_DoubleSpike_Barrel = cms.double( 1.0E9 ),
    thresh_Pt_Barrel = cms.double( 0.0 ),
    thresh_Clean_Barrel = cms.double( 80.0 ),
    PFRecHits = cms.InputTag( 'hltParticleFlowRecHitHCAL','HFEM' ),
    cleanRBXandHPDs = cms.bool( False ),
    depthCor_B = cms.double( 7.4 ),
    depthCor_A = cms.double( 0.89 ),
    nNeighbours = cms.int32( 0 ),
    thresh_DoubleSpike_Endcap = cms.double( 1.0E9 ),
    minS4S1_Clean_Barrel = cms.vdouble( 0.11, -0.19 ),
    thresh_Pt_Seed_Barrel = cms.double( 0.0 ),
    thresh_Pt_Endcap = cms.double( 0.0 ),
    thresh_Barrel = cms.double( 0.8 ),
    thresh_Clean_Endcap = cms.double( 80.0 ),
    thresh_Seed_Barrel = cms.double( 1.4 ),
    depthCor_Mode = cms.int32( 0 ),
    depthCor_B_preshower = cms.double( 4.0 ),
    useCornerCells = cms.bool( False ),
    depthCor_A_preshower = cms.double( 0.89 ),
    thresh_Endcap = cms.double( 0.8 ),
    thresh_Pt_Seed_Endcap = cms.double( 0.0 ),
    minS4S1_Clean_Endcap = cms.vdouble( 0.11, -0.19 ),
    thresh_Seed_Endcap = cms.double( 1.4 ),
    minS6S2_DoubleSpike_Endcap = cms.double( -1.0 ),
    minS6S2_DoubleSpike_Barrel = cms.double( -1.0 )
)
process.hltParticleFlowClusterHFHAD = cms.EDProducer( "PFClusterProducer",
    posCalcNCrystal = cms.int32( 5 ),
    verbose = cms.untracked.bool( False ),
    showerSigma = cms.double( 10.0 ),
    thresh_DoubleSpike_Barrel = cms.double( 1.0E9 ),
    thresh_Pt_Barrel = cms.double( 0.0 ),
    thresh_Clean_Barrel = cms.double( 120.0 ),
    PFRecHits = cms.InputTag( 'hltParticleFlowRecHitHCAL','HFHAD' ),
    cleanRBXandHPDs = cms.bool( False ),
    depthCor_B = cms.double( 7.4 ),
    depthCor_A = cms.double( 0.89 ),
    nNeighbours = cms.int32( 0 ),
    thresh_DoubleSpike_Endcap = cms.double( 1.0E9 ),
    minS4S1_Clean_Barrel = cms.vdouble( 0.045, -0.08 ),
    thresh_Pt_Seed_Barrel = cms.double( 0.0 ),
    thresh_Pt_Endcap = cms.double( 0.0 ),
    thresh_Barrel = cms.double( 0.8 ),
    thresh_Clean_Endcap = cms.double( 120.0 ),
    thresh_Seed_Barrel = cms.double( 1.4 ),
    depthCor_Mode = cms.int32( 0 ),
    depthCor_B_preshower = cms.double( 4.0 ),
    useCornerCells = cms.bool( False ),
    depthCor_A_preshower = cms.double( 0.89 ),
    thresh_Endcap = cms.double( 0.8 ),
    thresh_Pt_Seed_Endcap = cms.double( 0.0 ),
    minS4S1_Clean_Endcap = cms.vdouble( 0.045, -0.08 ),
    thresh_Seed_Endcap = cms.double( 1.4 ),
    minS6S2_DoubleSpike_Endcap = cms.double( -1.0 ),
    minS6S2_DoubleSpike_Barrel = cms.double( -1.0 )
)
process.hltParticleFlowClusterPS = cms.EDProducer( "PFClusterProducer",
    posCalcNCrystal = cms.int32( -1 ),
    verbose = cms.untracked.bool( False ),
    showerSigma = cms.double( 0.2 ),
    thresh_DoubleSpike_Barrel = cms.double( 1.0E9 ),
    thresh_Pt_Barrel = cms.double( 0.0 ),
    thresh_Clean_Barrel = cms.double( 100000.0 ),
    PFRecHits = cms.InputTag( "hltParticleFlowRecHitPS" ),
    cleanRBXandHPDs = cms.bool( False ),
    depthCor_B = cms.double( 7.4 ),
    depthCor_A = cms.double( 0.89 ),
    nNeighbours = cms.int32( 8 ),
    thresh_DoubleSpike_Endcap = cms.double( 1.0E9 ),
    minS4S1_Clean_Barrel = cms.vdouble( 0.0, 0.0 ),
    thresh_Pt_Seed_Barrel = cms.double( 0.0 ),
    thresh_Pt_Endcap = cms.double( 0.0 ),
    thresh_Barrel = cms.double( 6.0E-5 ),
    thresh_Clean_Endcap = cms.double( 100000.0 ),
    thresh_Seed_Barrel = cms.double( 1.2E-4 ),
    depthCor_Mode = cms.int32( 0 ),
    depthCor_B_preshower = cms.double( 4.0 ),
    useCornerCells = cms.bool( False ),
    depthCor_A_preshower = cms.double( 0.89 ),
    thresh_Endcap = cms.double( 6.0E-5 ),
    thresh_Pt_Seed_Endcap = cms.double( 0.0 ),
    minS4S1_Clean_Endcap = cms.vdouble( 0.0, 0.0 ),
    thresh_Seed_Endcap = cms.double( 1.2E-4 ),
    minS6S2_DoubleSpike_Endcap = cms.double( -1.0 ),
    minS6S2_DoubleSpike_Barrel = cms.double( -1.0 )
)
process.hltLightPFTracks = cms.EDProducer( "LightPFTrackProducer",
    TrackQuality = cms.string( "none" ),
    UseQuality = cms.bool( False ),
    TkColList = cms.VInputTag( 'hltPFMuonMerging' )
)
process.hltParticleFlowBlock = cms.EDProducer( "PFBlockProducer",
    PFClustersHCAL = cms.InputTag( "hltParticleFlowClusterHCAL" ),
    RecMuons = cms.InputTag( "hltMuons" ),
    PFClustersHFHAD = cms.InputTag( "hltParticleFlowClusterHFHAD" ),
    PFConversions = cms.InputTag( "" ),
    useConversions = cms.bool( False ),
    nuclearInteractionsPurity = cms.uint32( 1 ),
    PFClustersECAL = cms.InputTag( "hltParticleFlowClusterECAL" ),
    verbose = cms.untracked.bool( False ),
    PFClustersPS = cms.InputTag( "hltParticleFlowClusterPS" ),
    usePFatHLT = cms.bool( True ),
    PFClustersHO = cms.InputTag( "hltParticleFlowClusterHO" ),
    useIterTracking = cms.bool( False ),
    useConvBremPFRecTracks = cms.bool( False ),
    useV0 = cms.bool( False ),
    useNuclear = cms.bool( False ),
    EGPhotons = cms.InputTag( "" ),
    ConvBremGsfRecTracks = cms.InputTag( "" ),
    useKDTreeTrackEcalLinker = cms.bool( True ),
    useConvBremGsfTracks = cms.bool( False ),
    pf_DPtoverPt_Cut = cms.vdouble( 0.5, 0.5, 0.5, 0.5, 0.5 ),
    GsfRecTracks = cms.InputTag( "" ),
    RecTracks = cms.InputTag( "hltLightPFTracks" ),
    useHO = cms.bool( False ),
    PFNuclear = cms.InputTag( "" ),
    PFV0 = cms.InputTag( "" ),
    PhotonSelectionCuts = cms.vdouble(  ),
    PFClustersHFEM = cms.InputTag( "hltParticleFlowClusterHFEM" ),
    debug = cms.untracked.bool( False ),
    useEGPhotons = cms.bool( False ),
    pf_NHit_Cut = cms.vuint32( 3, 3, 3, 3, 3 )
)
process.hltParticleFlow = cms.EDProducer( "PFProducer",
    sumPtTrackIsoForEgammaSC_endcap = cms.double( 4.0 ),
    calibHF_use = cms.bool( False ),
    verbose = cms.untracked.bool( False ),
    minSignificance = cms.double( 2.5 ),
    usePhotonReg = cms.bool( False ),
    pf_nsigma_ECAL = cms.double( 0.0 ),
    usePFConversions = cms.bool( False ),
    useCalibrationsFromDB = cms.bool( True ),
    sumPtTrackIsoForPhoton = cms.double( -1.0 ),
    calibPFSCEle_endcap = cms.vdouble( 1.153, -16.5975, 5.668, -0.1772, 16.22, 7.326, 0.0483, -4.068, 9.406 ),
    usePFElectrons = cms.bool( False ),
    postMuonCleaning = cms.bool( True ),
    minDeltaMet = cms.double( 0.4 ),
    minSignificanceReduction = cms.double( 1.4 ),
    muon_HCAL = cms.vdouble( 3.0, 3.0 ),
    muon_HO = cms.vdouble( 0.9, 0.9 ),
    postHFCleaning = cms.bool( False ),
    factors_45 = cms.vdouble( 10.0, 100.0 ),
    cleanedHF = cms.VInputTag( 'hltParticleFlowRecHitHCAL:Cleaned','hltParticleFlowClusterHFHAD:Cleaned','hltParticleFlowClusterHFEM:Cleaned' ),
    iCfgCandConnector = cms.PSet( 
      bCalibSecondary = cms.bool( False ),
      bCalibPrimary = cms.bool( False ),
      bCorrect = cms.bool( False ),
      nuclCalibFactors = cms.vdouble( 0.8, 0.15, 0.5, 0.5, 0.05 )
    ),
    useBestMuonTrack = cms.bool( False ),
    rejectTracks_Bad = cms.bool( False ),
    coneEcalIsoForEgammaSC = cms.double( 0.3 ),
    usePFPhotons = cms.bool( False ),
    vertexCollection = cms.InputTag( "hltPixelVertices" ),
    sumPtTrackIsoForEgammaSC_barrel = cms.double( 4.0 ),
    egammaElectrons = cms.InputTag( "" ),
    calibHF_a_EMonly = cms.vdouble( 0.96945, 0.96701, 0.76309, 0.82268, 0.87583, 0.89718, 0.98674, 1.4681, 1.458, 1.458 ),
    maxDeltaPhiPt = cms.double( 7.0 ),
    muons = cms.InputTag( "hltMuons" ),
    pf_electronID_crackCorrection = cms.bool( False ),
    minHFCleaningPt = cms.double( 5.0 ),
    nTrackIsoForEgammaSC = cms.uint32( 2 ),
    pf_nsigma_HCAL = cms.double( 1.0 ),
    calibPFSCEle_Fbrem_barrel = cms.vdouble( 0.6, 6.0, -0.0255975, 0.0576727, 0.975442, -5.46394E-4, 1.26147, 25.0, -0.02025, 0.04537, 0.9728, -8.962E-4, 1.172 ),
    muon_ECAL = cms.vdouble( 0.5, 0.5 ),
    blocks = cms.InputTag( "hltParticleFlowBlock" ),
    calibPFSCEle_barrel = cms.vdouble( 1.004, -1.536, 22.88, -1.467, 0.3555, 0.6227, 14.65, 2051.0, 25.0, 0.9932, -0.5444, 0.0, 0.5438, 0.7109, 7.645, 0.2904, 0.0 ),
    pf_electron_mvaCut = cms.double( -0.1 ),
    useEGammaElectrons = cms.bool( False ),
    useHO = cms.bool( False ),
    nsigma_TRACK = cms.double( 1.0 ),
    pf_electron_output_col = cms.string( "electrons" ),
    dptRel_DispVtx = cms.double( 10.0 ),
    usePFMuonMomAssign = cms.bool( False ),
    useVerticesForNeutral = cms.bool( True ),
    pf_conv_mvaCut = cms.double( 0.0 ),
    sumEtEcalIsoForEgammaSC_endcap = cms.double( 2.0 ),
    pf_Res_mvaWeightFile = cms.string( "RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFRes.root" ),
    usePFDecays = cms.bool( False ),
    sumPtTrackIsoSlopeForPhoton = cms.double( -1.0 ),
    calibHF_b_EMHAD = cms.vdouble( 1.27541, 0.85361, 0.86333, 0.89091, 0.94348, 0.94348, 0.9437, 1.0034, 1.0444, 1.0444 ),
    rejectTracks_Step45 = cms.bool( False ),
    pf_GlobC_mvaWeightFile = cms.string( "RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFGlobalCorr.root" ),
    pf_locC_mvaWeightFile = cms.string( "RecoParticleFlow/PFProducer/data/TMVARegression_BDTG_PFClusterCorr.root" ),
    sumEtEcalIsoForEgammaSC_barrel = cms.double( 1.0 ),
    calibPFSCEle_Fbrem_endcap = cms.vdouble( 0.9, 6.5, -0.0692932, 0.101776, 0.995338, -0.00236548, 0.874998, 1.653, -0.0750184, 0.147, 0.923165, 4.74665E-4, 1.10782 ),
    coneTrackIsoForEgammaSC = cms.double( 0.3 ),
    usePFNuclearInteractions = cms.bool( False ),
    pf_electronID_mvaWeightFile = cms.string( "RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_PfElectrons23Jan_IntToFloat.txt" ),
    maxSignificance = cms.double( 2.5 ),
    calibHF_b_HADonly = cms.vdouble( 1.27541, 0.85361, 0.86333, 0.89091, 0.94348, 0.94348, 0.9437, 1.0034, 1.0444, 1.0444 ),
    calibHF_a_EMHAD = cms.vdouble( 1.42215, 1.00496, 0.68961, 0.81656, 0.98504, 0.98504, 1.00802, 1.0593, 1.4576, 1.4576 ),
    algoType = cms.uint32( 0 ),
    usePFSCEleCalib = cms.bool( True ),
    pt_Error = cms.double( 1.0 ),
    debug = cms.untracked.bool( False ),
    X0_Map = cms.string( "RecoParticleFlow/PFProducer/data/allX0histos.root" ),
    pf_convID_mvaWeightFile = cms.string( "RecoParticleFlow/PFProducer/data/MVAnalysis_BDT.weights_pfConversionAug0411.txt" ),
    calibHF_eta_step = cms.vdouble( 0.0, 2.9, 3.0, 3.2, 4.2, 4.4, 4.6, 4.8, 5.2, 5.4 ),
    useRegressionFromDB = cms.bool( False ),
    useEGammaSupercluster = cms.bool( False )
)
process.hltKT6PFJets = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 1 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( True ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "PFJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 5.0 ),
    doRhoFastjet = cms.bool( True ),
    jetAlgorithm = cms.string( "Kt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.6 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltParticleFlow" ),
    inputEtMin = cms.double( 0.0 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "hltPixelVertices" ),
    jetPtMin = cms.double( 0.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltAntiKT5PFJets = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( -9.0 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( True ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "PFJet" ),
    minSeed = cms.uint32( 0 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltParticleFlow" ),
    inputEtMin = cms.double( 0.0 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "hltPixelVertices" ),
    jetPtMin = cms.double( 0.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltAK5PFJetL1FastL2L3Corrected = cms.EDProducer( "PFJetCorrectionProducer",
    src = cms.InputTag( "hltAntiKT5PFJets" ),
    correctors = cms.vstring( 'hltESPAK5PFL1L2L3' )
)
process.hltPFJetsL1Matched = cms.EDProducer( "HLTPFJetL1MatchProducer",
    L1CenJets = cms.InputTag( 'hltL1extraParticles','Central' ),
    DeltaR = cms.double( 0.5 ),
    L1ForJets = cms.InputTag( 'hltL1extraParticles','Forward' ),
    L1TauJets = cms.InputTag( 'hltL1extraParticles','Tau' ),
    jetsInput = cms.InputTag( "hltAK5PFJetL1FastL2L3Corrected" )
)
process.hltPAMu3PFJet20 = cms.EDFilter( "HLT1PFJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPFJetsL1Matched" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPAL1sL1SingleMu3Jet36 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_Jet36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMu3PFJet40 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAMu3PFJet40L2Filtered3 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( False ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltPAL1sL1SingleMu3Jet36" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPAMu3PFJet40L3Filter3 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltPAMu3PFJet40L2Filtered3" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 3.0 )
)
process.hltPAMu3PFJet40 = cms.EDFilter( "HLT1PFJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPFJetsL1Matched" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleMu7Jet16 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu7_Jet16" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMu7PFJet20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAMu7PFJet20L2Filtered5 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( False ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu7Jet16" ),
    MinPt = cms.double( 5.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPAMu7PFJet20L3Filter3 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltPAMu7PFJet20L2Filtered5" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 7.0 )
)
process.hltPAMu7PFJet20 = cms.EDFilter( "HLT1PFJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPFJetsL1Matched" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleEG5BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAPhoton10NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGRegionalL1SingleEG5PA = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    saveTags = cms.bool( False ),
    endcap_end = cms.double( 2.65 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    barrel_end = cms.double( 1.4791 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candIsolatedTag = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    region_phi_size = cms.double( 1.044 ),
    region_eta_size = cms.double( 0.522 ),
    L1SeedFilterTag = cms.InputTag( "hltL1sL1SingleEG5BptxAND" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    candNonIsolatedTag = cms.InputTag( "" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
)
process.hltEG10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG5PA" ),
    etcutEB = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 10.0 )
)
process.hltPrePAPhoton15NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEG15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG5PA" ),
    etcutEB = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 15.0 )
)
process.hltPrePAPhoton20NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEG20EtPAFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG5PA" ),
    etcutEB = cms.double( 20.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 20.0 )
)
process.hltPrePAPhoton30NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAEG30EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG12" ),
    etcutEB = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 30.0 )
)
process.hltL1sL1SingleEG20 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG20" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAPhoton40NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGRegionalL1SingleEG20 = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    saveTags = cms.bool( False ),
    endcap_end = cms.double( 2.65 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    barrel_end = cms.double( 1.4791 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candIsolatedTag = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    region_phi_size = cms.double( 1.044 ),
    region_eta_size = cms.double( 0.522 ),
    L1SeedFilterTag = cms.InputTag( "hltL1sL1SingleEG20" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    candNonIsolatedTag = cms.InputTag( "" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
)
process.hltPAEG40EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG20" ),
    etcutEB = cms.double( 40.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 40.0 )
)
process.hltL1sL1SingleEG24 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG24" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAPhoton60NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGRegionalL1SingleEG24 = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    saveTags = cms.bool( False ),
    endcap_end = cms.double( 2.65 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    barrel_end = cms.double( 1.4791 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candIsolatedTag = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    region_phi_size = cms.double( 1.044 ),
    region_eta_size = cms.double( 0.522 ),
    L1SeedFilterTag = cms.InputTag( "hltL1sL1SingleEG24" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    candNonIsolatedTag = cms.InputTag( "" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
)
process.hltEG60EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG24" ),
    etcutEB = cms.double( 60.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 60.0 )
)
process.hltPrePAPhoton10TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEG10TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEG10EtFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltEG10TightCaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltEG10TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePAPhoton15TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEG15TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEG15EtFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltEG15TightCaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltEG15TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePAPhoton20TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEG20TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEG20EtPAFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltEG20TightCaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltEG20TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePAPhoton30TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAEG30TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltPAEG30EtFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPAEG30TightCaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltPAEG30TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePAPhoton40TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAEG40TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltPAEG40EtFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPAEG40TightCaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltPAEG40TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePAPhoton10TightCaloIdVLIso50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAPhoton10CaloIdVLIso50EcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.012 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.012 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonEcalIso" ),
    candTag = cms.InputTag( "hltEG10TightCaloIdVLHEFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPAPhoton10CaloIdVLIso50HcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.005 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.005 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalIso" ),
    candTag = cms.InputTag( "hltPAPhoton10CaloIdVLIso50EcalIsoFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPrePAPhoton15TightCaloIdVLIso50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAPhoton15CaloIdVLIso50EcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.012 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.012 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonEcalIso" ),
    candTag = cms.InputTag( "hltEG15TightCaloIdVLHEFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPAPhoton15CaloIdVLIso50HcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.005 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.005 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalIso" ),
    candTag = cms.InputTag( "hltPAPhoton15CaloIdVLIso50EcalIsoFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPrePAPhoton20TightCaloIdVLIso50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAPhoton20CaloIdVLIso50EcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.012 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.012 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonEcalIso" ),
    candTag = cms.InputTag( "hltEG20TightCaloIdVLHEFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPAPhoton20CaloIdVLIso50HcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.005 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.005 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalIso" ),
    candTag = cms.InputTag( "hltPAPhoton20CaloIdVLIso50EcalIsoFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPrePAPhoton30TightCaloIdVLIso50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAPhoton30CaloIdVLIso50EcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.012 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.012 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonEcalIso" ),
    candTag = cms.InputTag( "hltPAEG30TightCaloIdVLHEFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPAPhoton30CaloIdVLIso50HcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.005 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.005 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalIso" ),
    candTag = cms.InputTag( "hltPAPhoton30CaloIdVLIso50EcalIsoFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltL1sL1DoubleEG5 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAPhoton10Photon10NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGRegionalL1DoubleEG5 = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    saveTags = cms.bool( False ),
    endcap_end = cms.double( 2.65 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    barrel_end = cms.double( 1.4791 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candIsolatedTag = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    region_phi_size = cms.double( 1.044 ),
    region_eta_size = cms.double( 0.522 ),
    L1SeedFilterTag = cms.InputTag( "hltL1sL1DoubleEG5" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    candNonIsolatedTag = cms.InputTag( "" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
)
process.hltEGDouble10And10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 10.0 )
)
process.hltPrePAPhoton15Photon10NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGDouble15And10EtFilterFirst = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 10.0 )
)
process.hltEGDouble15And10EtFilterSecond = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 15.0 )
)
process.hltPrePAPhoton20Photon15NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGDouble20And15EtFilterFirst = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 15.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 15.0 )
)
process.hltEGDouble20And15EtFilterSecond = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 20.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 20.0 )
)
process.hltPrePAPhoton20Photon20NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGDouble20And20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 20.0 )
)
process.hltPrePAPhoton30Photon30NoCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGDouble30And30EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 30.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 30.0 )
)
process.hltPrePAPhoton10Photon10TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAEGDouble10And10TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEGDouble10And10EtFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPAEGDouble10And10CaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltPAEGDouble10And10TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePAPhoton10Photon10TightCaloIdVLIso50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAPhoton10AndPhoton10CaloIdVLIso50EcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.012 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.012 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonEcalIso" ),
    candTag = cms.InputTag( "hltPAEGDouble10And10CaloIdVLHEFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPAPhoton10AndPhoton10CaloIdVLIso50HcalIsoFilter = cms.EDFilter( "HLTEgammaGenericQuadraticFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 5.0 ),
    thrOverEEE = cms.double( 0.005 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.005 ),
    thrRegularEB = cms.double( 5.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalIso" ),
    candTag = cms.InputTag( "hltPAPhoton10AndPhoton10CaloIdVLIso50EcalIsoFilter" ),
    thrOverE2EE = cms.double( 0.0 )
)
process.hltPrePAPhoton15Photon10TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAEGDouble15And10TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEGDouble15And10EtFilterSecond" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPAEGDouble15And10CaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltPAEGDouble15And10TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePAPhoton20Photon15TightCaloIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAEGDouble20And15TightCaloIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.035 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.014 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEGDouble20And15EtFilterSecond" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPAEGDouble20And15CaloIdVLHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( True ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltPAEGDouble20And15TightCaloIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltPrePASingleEle6CaloIdTTrkIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleEG6EtFilterL1SingleEG5 = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG5PA" ),
    etcutEB = cms.double( 6.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 6.0 )
)
process.hltSingleEle6CaloIdTTrlIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.031 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.011 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltSingleEG6EtFilterL1SingleEG5" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltSingleEle6CaloIdTHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.075 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.1 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltSingleEle6CaloIdTTrlIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltSingleEle6CaloIdTPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltSingleEle6CaloIdTHEFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltCkfL1SeededTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPCkfTrajectoryBuilder" )
)
process.hltCtfL1SeededWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltCkfL1SeededTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltPixelMatchElectronsL1Seeded = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    BSProducer = cms.InputTag( "hltOnlineBeamSpot" ),
    UseGsfTracks = cms.bool( False ),
    TrackProducer = cms.InputTag( "hltCtfL1SeededWithMaterialTracks" ),
    GsfTrackProducer = cms.InputTag( "" )
)
process.hltEle6CaloIdTOneOEMinusOneOPSingleFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    electronNonIsolatedProducer = cms.InputTag( "" ),
    barrelcut = cms.double( 999.9 ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltSingleEle6CaloIdTPixelMatchFilter" ),
    endcapcut = cms.double( 999.9 )
)
process.hltElectronL1SeededDetaDphi = cms.EDProducer( "EgammaHLTElectronDetaDphiProducer",
    variablesAtVtx = cms.bool( False ),
    useSCRefs = cms.bool( False ),
    BSProducer = cms.InputTag( "hltOnlineBeamSpot" ),
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    recoEcalCandidateProducer = cms.InputTag( "" ),
    useTrackProjectionToEcal = cms.bool( False )
)
process.hltEle6CaloIdTTrkIdVLDetaSingleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( False ),
    thrRegularEE = cms.double( 0.01 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.01 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Deta' ),
    candTag = cms.InputTag( "hltEle6CaloIdTOneOEMinusOneOPSingleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltEle6CaloIdTTrkIdVLDphiSingleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    thrRegularEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.15 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Dphi' ),
    candTag = cms.InputTag( "hltEle6CaloIdTTrkIdVLDetaSingleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltPrePASingleEle6CaloIdNoneTrkIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSingleEle6CaloIdNoneTrlIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 999.0 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 999.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltSingleEG6EtFilterL1SingleEG5" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltSingleEle6CaloIdNoneHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 999.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 999.0 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltSingleEle6CaloIdNoneTrlIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltSingleEle6CaloIdNonePixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltSingleEle6CaloIdNoneHEFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltEle6CaloIdNoneOneOEMinusOneOPSingleFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    electronNonIsolatedProducer = cms.InputTag( "" ),
    barrelcut = cms.double( 999.9 ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltSingleEle6CaloIdNonePixelMatchFilter" ),
    endcapcut = cms.double( 999.9 )
)
process.hltEle6CaloIdNoneTrkIdVLDetaSingleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( False ),
    thrRegularEE = cms.double( 999.0 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 999.0 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Deta' ),
    candTag = cms.InputTag( "hltEle6CaloIdNoneOneOEMinusOneOPSingleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltEle6CaloIdNoneTrkIdVLDphiSingleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    thrRegularEE = cms.double( 999.0 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 999.0 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Dphi' ),
    candTag = cms.InputTag( "hltEle6CaloIdNoneTrkIdVLDetaSingleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltL1sL1SingleEG7 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG7" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePASingleEle8CaloIdNoneTrkIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEGRegionalL1SingleEG7 = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    saveTags = cms.bool( False ),
    endcap_end = cms.double( 2.65 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    barrel_end = cms.double( 1.4791 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candIsolatedTag = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    region_phi_size = cms.double( 1.044 ),
    region_eta_size = cms.double( 0.522 ),
    L1SeedFilterTag = cms.InputTag( "hltL1sL1SingleEG7" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    candNonIsolatedTag = cms.InputTag( "" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
)
process.hltSingleEG8EtFilterL1SingleEG7 = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG7" ),
    etcutEB = cms.double( 8.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 8.0 )
)
process.hltSingleEle8CaloIdNoneTrlIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 999.0 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 999.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltSingleEG8EtFilterL1SingleEG7" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltSingleEle8CaloIdNoneHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 999.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 999.0 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltSingleEle8CaloIdNoneTrlIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltSingleEle8CaloIdNonePixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltSingleEle8CaloIdNoneHEFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltEle8CaloIdNoneOneOEMinusOneOPSingleFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    electronNonIsolatedProducer = cms.InputTag( "" ),
    barrelcut = cms.double( 999.9 ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltSingleEle8CaloIdNonePixelMatchFilter" ),
    endcapcut = cms.double( 999.9 )
)
process.hltEle8CaloIdNoneTrkIdVLDetaSingleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( False ),
    thrRegularEE = cms.double( 999.0 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 999.0 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Deta' ),
    candTag = cms.InputTag( "hltEle8CaloIdNoneOneOEMinusOneOPSingleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltEle8CaloIdNoneTrkIdVLDphiSingleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    thrRegularEE = cms.double( 999.0 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 999.0 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Dphi' ),
    candTag = cms.InputTag( "hltEle8CaloIdNoneTrkIdVLDetaSingleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltPrePAL1DoubleEG5DoubleEle6CaloIdTTrkIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltDoubleEG5DoubleEle6EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1DoubleEG5" ),
    etcutEB = cms.double( 6.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 6.0 )
)
process.hltDoubleEG5DoubleEle6CaloIdTTrlIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.031 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.011 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltDoubleEG5DoubleEle6EtFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltDoubleEG5DoubleEle6CaloIdTHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.075 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.1 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltDoubleEG5DoubleEle6CaloIdTTrlIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltDoubleEG5DoubleEle6CaloIdTPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltDoubleEG5DoubleEle6CaloIdTHEFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltDoubleEG5DoubleEle6CaloIdTOneOEMinusOneOPDoubleFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    electronNonIsolatedProducer = cms.InputTag( "" ),
    barrelcut = cms.double( 999.9 ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltDoubleEG5DoubleEle6CaloIdTPixelMatchFilter" ),
    endcapcut = cms.double( 999.9 )
)
process.hltDoubleEG5DoubleEle6CaloIdTTrkIdVLDetaDoubleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( False ),
    thrRegularEE = cms.double( 0.01 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.01 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Deta' ),
    candTag = cms.InputTag( "hltDoubleEG5DoubleEle6CaloIdTOneOEMinusOneOPDoubleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltDoubleEG5DoubleEle6CaloIdTTrkIdVLDphiDoubleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    thrRegularEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.15 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Dphi' ),
    candTag = cms.InputTag( "hltDoubleEG5DoubleEle6CaloIdTTrkIdVLDetaDoubleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltPrePADoubleEle6CaloIdTTrkIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltDoubleEG6EtFilterL1SingleEG5 = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG5PA" ),
    etcutEB = cms.double( 6.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 6.0 )
)
process.hltDoubleEle6CaloIdTTrlIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.031 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.011 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltDoubleEG6EtFilterL1SingleEG5" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltDoubleEle6CaloIdTHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.075 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.1 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltDoubleEle6CaloIdTTrlIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltDoubleEle6CaloIdTPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltDoubleEle6CaloIdTHEFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltEle6CaloIdTOneOEMinusOneOPDoubleFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    electronNonIsolatedProducer = cms.InputTag( "" ),
    barrelcut = cms.double( 999.9 ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltDoubleEle6CaloIdTPixelMatchFilter" ),
    endcapcut = cms.double( 999.9 )
)
process.hltEle6CaloIdTTrkIdVLDetaDoubleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( False ),
    thrRegularEE = cms.double( 0.01 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.01 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Deta' ),
    candTag = cms.InputTag( "hltEle6CaloIdTOneOEMinusOneOPDoubleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltEle6CaloIdTTrkIdVLDphiDoubleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    thrRegularEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.15 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Dphi' ),
    candTag = cms.InputTag( "hltEle6CaloIdTTrkIdVLDetaDoubleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltPrePADoubleEle8CaloIdTTrkIdVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltDoubleEG8EtFilterL1SingleEG7 = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1SingleEG7" ),
    etcutEB = cms.double( 8.0 ),
    ncandcut = cms.int32( 2 ),
    etcutEE = cms.double( 8.0 )
)
process.hltDoubleEle8CaloIdTTrlIdVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.031 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.011 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltDoubleEG8EtFilterL1SingleEG7" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltDoubleEle8CaloIdTHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.075 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.1 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltDoubleEle8CaloIdTTrlIdVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltDoubleEle8CaloIdTPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltDoubleEle8CaloIdTHEFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltEle8CaloIdTOneOEMinusOneOPDoubleFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    saveTags = cms.bool( False ),
    doIsolated = cms.bool( True ),
    electronNonIsolatedProducer = cms.InputTag( "" ),
    barrelcut = cms.double( 999.9 ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltDoubleEle8CaloIdTPixelMatchFilter" ),
    endcapcut = cms.double( 999.9 )
)
process.hltEle8CaloIdTTrkIdVLDetaDoubleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( False ),
    thrRegularEE = cms.double( 0.01 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.01 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Deta' ),
    candTag = cms.InputTag( "hltEle8CaloIdTOneOEMinusOneOPDoubleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltEle8CaloIdTTrkIdVLDphiDoubleFilter = cms.EDFilter( "HLTElectronGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    thrTimesPtEB = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    thrRegularEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Seeded" ),
    thrRegularEB = cms.double( 0.15 ),
    lessThan = cms.bool( True ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( 'hltElectronL1SeededDetaDphi','Dphi' ),
    candTag = cms.InputTag( "hltEle8CaloIdTTrkIdVLDetaDoubleFilter" ),
    thrTimesPtEE = cms.double( -1.0 ),
    thrOverPtEE = cms.double( -1.0 ),
    thrOverPtEB = cms.double( -1.0 )
)
process.hltL1sETT20BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT20_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAPixelTracksMultiplicity100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHcalPM2Tower3GeVFilter = cms.EDFilter( "HLTHcalTowerFilter",
    saveTags = cms.bool( False ),
    MinN_HFM = cms.int32( 2 ),
    MinE_HB = cms.double( -1.0 ),
    MaxN_HB = cms.int32( 999999999 ),
    inputTag = cms.InputTag( "hltTowerMakerForHf" ),
    MaxN_HE = cms.int32( 999999999 ),
    MinE_HE = cms.double( -1.0 ),
    MinE_HF = cms.double( 3.0 ),
    MinN_HF = cms.int32( 4 ),
    MaxN_HF = cms.int32( 999999999 ),
    MinN_HFP = cms.int32( 2 )
)
process.hltPAPixelTracksForHighMult = cms.EDProducer( "PixelTrackProducer",
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      ptMin = cms.double( 0.3 ),
      tipMax = cms.double( 1.0 )
    ),
    useFilterWithES = cms.bool( False ),
    passLabel = cms.string( "Pixel triplet tracks for vertexing" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      fixImpactParameter = cms.double( 0.0 )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originHalfLength = cms.double( 15.1 ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 0.3 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "hltESPPixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.06 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ) )
      )
    )
)
process.hltPAPixelVerticesForHighMult = cms.EDProducer( "PrimaryVertexProducer",
    vertexCollections = cms.VPSet( 
      cms.PSet(  maxDistanceToBeam = cms.double( 2.0 ),
        useBeamConstraint = cms.bool( False ),
        minNdof = cms.double( 0.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" ),
        label = cms.string( "" )
      )
    ),
    verbose = cms.untracked.bool( False ),
    TkFilterParameters = cms.PSet( 
      maxD0Significance = cms.double( 100.0 ),
      minPt = cms.double( 0.3 ),
      maxNormalizedChi2 = cms.double( 100.0 ),
      minSiliconLayersWithHits = cms.int32( 3 ),
      minPixelLayersWithHits = cms.int32( 3 ),
      trackQuality = cms.string( "any" ),
      algorithm = cms.string( "filter" )
    ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    TrackLabel = cms.InputTag( "hltPAPixelTracksForHighMult" ),
    TkClusParameters = cms.PSet( 
      algorithm = cms.string( "gap" ),
      TkGapClusParameters = cms.PSet(  zSeparation = cms.double( 0.1 ) )
    )
)
process.hltPAGoodPixelTracksForHighMult = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "loose" ),
    minNumberLayers = cms.uint32( 0 ),
    chi2n_par = cms.double( 9999.0 ),
    useVtxError = cms.bool( True ),
    nSigmaZ = cms.double( 100.0 ),
    dz_par2 = cms.vdouble( 4.0, 0.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 9999.0, 0.0 ),
    copyTrajectories = cms.untracked.bool( False ),
    vtxNumber = cms.int32( 1 ),
    max_d0NoPV = cms.double( 0.5 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 999 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 999.0 ),
    copyExtras = cms.untracked.bool( False ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAPixelTracksForHighMult" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 9999.0, 0.0 ),
    d0_par1 = cms.vdouble( 9999.0, 0.0 ),
    res_par = cms.vdouble( 99999.0, 99999.0 ),
    minHitsToBypassChecks = cms.uint32( 999 )
)
process.hltPAPixelCandsForHighMult = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPAGoodPixelTracksForHighMult" ),
    particleType = cms.string( "pi+" )
)
process.hlt1PAHighMult100 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 100 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAPixelTracksMultiplicity130 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult130 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 130 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltL1sETT40 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT40" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAPixelTracksMultiplicity160 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult160 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 160 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAPixelTracksMultiplicity190 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult190 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 190 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltL1sETT60 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT60" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAPixelTracksMultiplicity220 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult220 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 220 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAPixelTrackMultiplicity100FullTrack12 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAPixelTracksForHighPt = cms.EDProducer( "PixelTrackProducer",
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      ptMin = cms.double( 0.0 ),
      tipMax = cms.double( 1.0 )
    ),
    useFilterWithES = cms.bool( False ),
    passLabel = cms.string( "Pixel triplet tracks for vertexing" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      fixImpactParameter = cms.double( 0.0 )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originHalfLength = cms.double( 15.1 ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 6.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "hltESPPixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.06 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ) )
      )
    )
)
process.hltPAPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 0.6 ),
    useProtoTrackKinematics = cms.bool( False ),
    InputVertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    InputCollection = cms.InputTag( "hltPAPixelTracksForHighPt" ),
    originRadius = cms.double( 0.2 )
)
process.hltPACkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltPAPixelSeedsFromPixelTracks" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPTrajectoryBuilderIT" )
)
process.hltPACtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltPACkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter0" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltPATrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.4, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.35, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPACtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.4, 4.0 ),
    d0_par1 = cms.vdouble( 0.3, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltPATrackRefsForJetsIter0 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltPATrackSelectionHighPurity" ),
    particleType = cms.string( "pi+" )
)
process.hltPAAntiKT5TrackJetsIter0 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltPATrackRefsForJetsIter0" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    jetPtMin = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltPATrackAndTauJetsIter0 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 1.0 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltPAAntiKT5TrackJetsIter0" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.7 ),
    etaMaxCaloJet = cms.double( 2.7 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltPATrackSelectionHighPurity" )
)
process.hltPAIter1ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltPATrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 9.0 ) ),
    doPixel = cms.bool( True )
)
process.hltPAIter1SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter1ESPMeasurementTrackerPA" )
)
process.hltPAIter1PixelSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 1.0 ),
        originHalfLength = cms.double( 0.2 ),
        originRadius = cms.double( 0.1 ),
        measurementTrackerName = cms.string( "hltIter1ESPMeasurementTrackerPA" ),
        deltaEtaRegion = cms.double( 1.0 ),
        vertexSrc = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltPATrackAndTauJetsIter0" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 6.0 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 100000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter1ESPPixelLayerTripletsPA" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltPAIter1CkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltPAIter1PixelSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter1ESPTrajectoryBuilderITPA" )
)
process.hltPAIter1CtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltPAIter1CkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter1" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltPAIter1TrackSelectionHighPurityLoose = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.9, 3.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.8, 3.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAIter1CtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.9, 3.0 ),
    d0_par1 = cms.vdouble( 0.85, 3.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltPAIter1TrackSelectionHighPurityTight = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 5 ),
    chi2n_par = cms.double( 0.4 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 1.0, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 1.0, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAIter1CtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 1.0, 4.0 ),
    d0_par1 = cms.vdouble( 1.0, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltPAIter1TrackSelectionHighPurity = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltPAIter1TrackSelectionHighPurityLoose" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltPAIter1TrackSelectionHighPurityTight" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltPAIter1Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltPATrackSelectionHighPurity" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltPAIter1TrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltPATrackRefsForJetsIter1 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltPAIter1Merged" ),
    particleType = cms.string( "pi+" )
)
process.hltPAAntiKT5TrackJetsIter1 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltPATrackRefsForJetsIter1" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    jetPtMin = cms.double( 1.4 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltPATrackAndTauJetsIter1 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 1.4 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltPAAntiKT5TrackJetsIter1" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.7 ),
    etaMaxCaloJet = cms.double( 2.7 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltPAIter1Merged" )
)
process.hltPAIter2ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltPAIter1TrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "hltPAIter1ClustersRefRemoval" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 16.0 ) ),
    doPixel = cms.bool( True )
)
process.hltPAIter2SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter2ESPMeasurementTrackerPA" )
)
process.hltPAIter2PixelSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 0.8 ),
        originHalfLength = cms.double( 0.1 ),
        originRadius = cms.double( 0.05 ),
        measurementTrackerName = cms.string( "hltIter2ESPMeasurementTrackerPA" ),
        deltaEtaRegion = cms.double( 0.8 ),
        vertexSrc = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltPATrackAndTauJetsIter1" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 6.0 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      GeneratorPSet = cms.PSet( 
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter2ESPPixelLayerPairsPA" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltPAIter2CkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltPAIter2PixelSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter2ESPTrajectoryBuilderITPA" )
)
process.hltPAIter2CtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltPAIter2CkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter2" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltPAIter2TrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.4, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.35, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAIter2CtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.4, 4.0 ),
    d0_par1 = cms.vdouble( 0.3, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltPAIter2Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltPAIter1Merged" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltPAIter2TrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltPATrackRefsForJetsIter2 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltPAIter2Merged" ),
    particleType = cms.string( "pi+" )
)
process.hltPAAntiKT5TrackJetsIter2 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltPATrackRefsForJetsIter2" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    jetPtMin = cms.double( 3.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltPATrackAndTauJetsIter2 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 3.0 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltPAAntiKT5TrackJetsIter2" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.7 ),
    etaMaxCaloJet = cms.double( 2.7 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltPAIter2Merged" )
)
process.hltPAIter3ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltPAIter2TrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "hltPAIter2ClustersRefRemoval" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 16.0 ) ),
    doPixel = cms.bool( True )
)
process.hltPAIter3SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter3ESPMeasurementTrackerPA" )
)
process.hltPAIter3MixedSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 0.5 ),
        originHalfLength = cms.double( 0.1 ),
        originRadius = cms.double( 0.1 ),
        measurementTrackerName = cms.string( "hltIter3ESPMeasurementTrackerPA" ),
        deltaEtaRegion = cms.double( 0.5 ),
        vertexSrc = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltPATrackAndTauJetsIter2" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 6.0 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 100000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter3ESPLayerTripletsPA" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsTripletOnlyCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltPAIter3CkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltPAIter3MixedSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter3ESPTrajectoryBuilderITPA" )
)
process.hltPAIter3CtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltPAIter3CkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter3" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltPAIter3TrackSelectionHighPurityLoose = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 3 ),
    chi2n_par = cms.double( 0.7 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 0.9, 3.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 0.85, 3.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAIter3CtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 0.9, 3.0 ),
    d0_par1 = cms.vdouble( 0.85, 3.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltPAIter3TrackSelectionHighPurityTight = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 5 ),
    chi2n_par = cms.double( 0.4 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 1.0, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 1.0, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 1 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAIter3CtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 1.0, 4.0 ),
    d0_par1 = cms.vdouble( 1.0, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltPAIter3TrackSelectionHighPurity = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltPAIter3TrackSelectionHighPurityLoose" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltPAIter3TrackSelectionHighPurityTight" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltPAIter3Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltPAIter2Merged" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltPAIter3TrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltPATrackRefsForJetsIter3 = cms.EDProducer( "ChargedRefCandidateProducer",
    src = cms.InputTag( "hltPAIter3Merged" ),
    particleType = cms.string( "pi+" )
)
process.hltPAAntiKT5TrackJetsIter3 = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( 0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "TrackJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltPATrackRefsForJetsIter3" ),
    inputEtMin = cms.double( 0.1 ),
    puPtMin = cms.double( 0.0 ),
    srcPVs = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    jetPtMin = cms.double( 4.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 0 ),
    MaxVtxZ = cms.double( 30.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( True ),
    DzTrVtxMax = cms.double( 0.5 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.2 )
)
process.hltPATrackAndTauJetsIter3 = cms.EDProducer( "TauJetSelectorForHLTTrackSeeding",
    fractionMinCaloInTauCone = cms.double( 0.7 ),
    fractionMaxChargedPUInCaloCone = cms.double( 0.3 ),
    tauConeSize = cms.double( 0.2 ),
    ptTrkMaxInCaloCone = cms.double( 4.0 ),
    isolationConeSize = cms.double( 0.5 ),
    inputTrackJetTag = cms.InputTag( "hltPAAntiKT5TrackJetsIter3" ),
    nTrkMaxInCaloCone = cms.int32( 0 ),
    inputCaloJetTag = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
    etaMinCaloJet = cms.double( -2.7 ),
    etaMaxCaloJet = cms.double( 2.7 ),
    ptMinCaloJet = cms.double( 5.0 ),
    inputTrackTag = cms.InputTag( "hltPAIter3Merged" )
)
process.hltPAIter4ClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemover",
    doStrip = cms.bool( True ),
    trajectories = cms.InputTag( "hltPAIter3TrackSelectionHighPurity" ),
    oldClusterRemovalInfo = cms.InputTag( "hltPAIter3ClustersRefRemoval" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    Common = cms.PSet(  maxChi2 = cms.double( 16.0 ) ),
    doPixel = cms.bool( True )
)
process.hltPAIter4SiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltIter4ESPMeasurementTrackerPA" )
)
process.hltPAIter4PixelLessSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        deltaPhiRegion = cms.double( 0.5 ),
        originHalfLength = cms.double( 2.0 ),
        originRadius = cms.double( 1.0 ),
        measurementTrackerName = cms.string( "hltIter4ESPMeasurementTrackerPA" ),
        deltaEtaRegion = cms.double( 0.5 ),
        vertexSrc = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
        searchOpt = cms.bool( True ),
        JetSrc = cms.InputTag( "hltPATrackAndTauJetsIter3" ),
        originZPos = cms.double( 0.0 ),
        ptMin = cms.double( 6.0 )
      )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
      doClusterCheck = cms.bool( False ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32( 10000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 0 ),
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      GeneratorPSet = cms.PSet( 
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltIter4ESPPixelLayerPairsPA" )
    ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      propagator = cms.string( "PropagatorWithMaterial" )
    ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
process.hltPAIter4CkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltPAIter4PixelLessSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltIter4ESPTrajectoryBuilderITPA" )
)
process.hltPAIter4CtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltPAIter4CkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "iter4" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltPAIter4TrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 5 ),
    chi2n_par = cms.double( 0.25 ),
    useVtxError = cms.bool( False ),
    nSigmaZ = cms.double( 3.0 ),
    dz_par2 = cms.vdouble( 1.0, 4.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 1.0, 4.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 0 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 9999.0 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=3" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAIter4CtfWithMaterialTracks" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 1.0, 4.0 ),
    d0_par1 = cms.vdouble( 1.0, 4.0 ),
    res_par = cms.vdouble( 0.003, 0.001 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltPAIter4Merged = cms.EDProducer( "SimpleTrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    promoteTrackQuality = cms.bool( True ),
    MinPT = cms.double( 0.05 ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    allowFirstHitShare = cms.bool( True ),
    newQuality = cms.string( "confirmed" ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    TrackProducer1 = cms.string( "hltPAIter3Merged" ),
    MinFound = cms.int32( 3 ),
    TrackProducer2 = cms.string( "hltPAIter4TrackSelectionHighPurity" ),
    LostHitPenalty = cms.double( 20.0 ),
    FoundHitBonus = cms.double( 5.0 )
)
process.hltPAGoodFullTracks = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "loose" ),
    minNumberLayers = cms.uint32( 0 ),
    chi2n_par = cms.double( 9999.0 ),
    useVtxError = cms.bool( True ),
    nSigmaZ = cms.double( 100.0 ),
    dz_par2 = cms.vdouble( 4.0, 0.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 9999.0, 0.0 ),
    copyTrajectories = cms.untracked.bool( False ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 0.5 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 999 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 0.15 ),
    copyExtras = cms.untracked.bool( False ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "tracksSize>=2" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 0 ),
    src = cms.InputTag( "hltPAIter4Merged" ),
    chi2n_no1Dmod_par = cms.double( 9999.0 ),
    vertices = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 4.0, 0.0 ),
    d0_par1 = cms.vdouble( 9999.0, 0.0 ),
    res_par = cms.vdouble( 99999.0, 99999.0 ),
    minHitsToBypassChecks = cms.uint32( 999 )
)
process.hltPAFullCands = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPAGoodFullTracks" ),
    particleType = cms.string( "pi+" )
)
process.hlt1PAFullTrack12 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 12.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAFullCands" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAPixelTrackMultiplicity130FullTrack12 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAPixelTrackMultiplicity160FullTrack12 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet12BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet12_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAFullTrack12 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAFullTrack20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAFullTrack20 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 20.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAFullCands" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAFullTrack30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAFullTrack30 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 30.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAFullCands" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAFullTrack50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAFullTrack50 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 50.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAFullCands" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAPixelTrackMultiplicity140Jet80NoJetID = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult140 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 140 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAPixelTrackMultiplicity100L2DoubleMu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePPPixelTracksMultiplicity55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult55 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 55 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePPPixelTracksMultiplicity70 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult70 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 70 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePPPixelTracksMultiplicity85 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hlt1PAHighMult85 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 85 ),
    MinPt = cms.double( 0.4 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHighMult" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePPPixelTrackMultiplicity55FullTrack12 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePPPixelTrackMultiplicity70FullTrack12 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleJetC36 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleJetC36" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePPL1DoubleJetC36 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sPATech35 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "35" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPrePATech35 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePATech35HFSumET100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAMetForHf = cms.EDProducer( "METProducer",
    resolutionsEra = cms.string( "Spring10" ),
    HB_EtResPar = cms.vdouble( 0.0, 1.22, 0.05 ),
    EE_PhiResPar = cms.vdouble( 0.02511 ),
    jdpt9 = cms.vdouble( 0.843, 0.885, 1.245, 1.665, 1.944, 1.981, 1.972, 2.875, 3.923, 7.51 ),
    jdpt8 = cms.vdouble( 0.889, 0.939, 1.166, 1.365, 1.553, 1.805, 2.06, 2.22, 2.268, 2.247 ),
    jdpt7 = cms.vdouble( 1.094, 1.139, 1.436, 1.672, 1.831, 2.05, 2.267, 2.549, 2.785, 2.86 ),
    jdpt6 = cms.vdouble( 1.213, 1.298, 1.716, 2.015, 2.191, 2.612, 2.863, 2.879, 2.925, 2.902 ),
    jdpt5 = cms.vdouble( 1.049, 1.149, 1.607, 1.869, 2.012, 2.219, 2.289, 2.412, 2.695, 2.865 ),
    jdpt4 = cms.vdouble( 0.85, 0.961, 1.337, 1.593, 1.854, 2.005, 2.209, 2.533, 2.812, 3.047 ),
    jdpt3 = cms.vdouble( 0.929, 1.04, 1.46, 1.74, 2.042, 2.289, 2.639, 2.837, 2.946, 2.971 ),
    jdpt2 = cms.vdouble( 0.841, 0.937, 1.316, 1.605, 1.919, 2.295, 2.562, 2.722, 2.943, 3.293 ),
    jdpt1 = cms.vdouble( 0.718, 0.813, 1.133, 1.384, 1.588, 1.841, 2.115, 2.379, 2.508, 2.772 ),
    jdpt0 = cms.vdouble( 0.749, 0.829, 1.099, 1.355, 1.584, 1.807, 2.035, 2.217, 2.378, 2.591 ),
    HE_EtResPar = cms.vdouble( 0.0, 1.3, 0.05 ),
    alias = cms.string( "RawCaloMET" ),
    HF_PhiResPar = cms.vdouble( 0.05022 ),
    InputType = cms.string( "CandidateCollection" ),
    HE_PhiResPar = cms.vdouble( 0.02511 ),
    HB_PhiResPar = cms.vdouble( 0.02511 ),
    EE_EtResPar = cms.vdouble( 0.2, 0.03, 0.005 ),
    noHF = cms.bool( False ),
    PF_PhiResType2 = cms.vdouble( 0.002 ),
    PF_PhiResType3 = cms.vdouble( 0.002 ),
    HF_EtResPar = cms.vdouble( 0.0, 1.82, 0.09 ),
    resolutionsAlgo = cms.string( "AK5PF" ),
    PF_PhiResType6 = cms.vdouble( 0.02511 ),
    PF_PhiResType7 = cms.vdouble( 0.02511 ),
    PF_PhiResType4 = cms.vdouble( 0.0028, 0.0, 0.0022 ),
    PF_PhiResType5 = cms.vdouble( 0.1, 0.1, 0.13 ),
    ptresolthreshold = cms.double( 10.0 ),
    METType = cms.string( "CaloMET" ),
    EB_EtResPar = cms.vdouble( 0.2, 0.03, 0.005 ),
    PF_PhiResType1 = cms.vdouble( 0.002 ),
    globalThreshold = cms.double( 0.0 ),
    EB_PhiResPar = cms.vdouble( 0.00502 ),
    src = cms.InputTag( "hltTowerMakerForHf" ),
    jdphi9 = cms.vdouble( 0.062, 0.059, 0.053, 0.047, 0.042, 0.045, 0.036, 0.032, 0.034, 0.044 ),
    jdphi8 = cms.vdouble( 0.059, 0.057, 0.051, 0.044, 0.038, 0.035, 0.037, 0.032, 0.028, 0.028 ),
    jdphi4 = cms.vdouble( 0.042, 0.042, 0.043, 0.042, 0.038, 0.036, 0.036, 0.033, 0.031, 0.031 ),
    jdphi3 = cms.vdouble( 0.042, 0.043, 0.044, 0.043, 0.041, 0.039, 0.039, 0.036, 0.034, 0.031 ),
    jdphi2 = cms.vdouble( 0.04, 0.04, 0.04, 0.04, 0.04, 0.038, 0.036, 0.035, 0.034, 0.033 ),
    jdphi1 = cms.vdouble( 0.034, 0.035, 0.035, 0.035, 0.035, 0.034, 0.031, 0.03, 0.029, 0.027 ),
    jdphi0 = cms.vdouble( 0.034, 0.034, 0.034, 0.034, 0.032, 0.031, 0.028, 0.027, 0.027, 0.027 ),
    jdphi7 = cms.vdouble( 0.077, 0.072, 0.059, 0.05, 0.045, 0.042, 0.039, 0.039, 0.037, 0.031 ),
    jdphi6 = cms.vdouble( 0.084, 0.08, 0.072, 0.065, 0.066, 0.06, 0.051, 0.049, 0.045, 0.045 ),
    jdphi5 = cms.vdouble( 0.069, 0.069, 0.064, 0.058, 0.053, 0.049, 0.049, 0.043, 0.039, 0.04 ),
    HO_EtResPar = cms.vdouble( 0.0, 1.3, 0.005 ),
    HO_PhiResPar = cms.vdouble( 0.02511 ),
    PF_EtResType5 = cms.vdouble( 0.41, 0.52, 0.25 ),
    PF_EtResType4 = cms.vdouble( 0.042, 0.1, 0.0 ),
    PF_EtResType7 = cms.vdouble( 0.0, 1.22, 0.05 ),
    PF_EtResType6 = cms.vdouble( 0.0, 1.22, 0.05 ),
    PF_EtResType1 = cms.vdouble( 0.05, 0.0, 0.0 ),
    calculateSignificance = cms.bool( False ),
    PF_EtResType3 = cms.vdouble( 0.05, 0.0, 0.0 ),
    PF_EtResType2 = cms.vdouble( 0.05, 0.0, 0.0 ),
    usePt = cms.untracked.bool( True ),
    onlyFiducialParticles = cms.bool( False ),
    vertexRho = cms.double( 2.0 ),
    eVetoDeltaPhi = cms.double( 100.0 ),
    PFClustersHCAL = cms.InputTag( "particleFlowClusterHCAL" ),
    PFClustersHFHAD = cms.InputTag( "particleFlowClusterHFHAD" ),
    dupMinPt = cms.double( 0.0 ),
    chi2_tight_max = cms.double( 5.0 ),
    vertexZ = cms.double( 15.0 ),
    nLayersTight = cms.int32( 0 ),
    vertexNdof = cms.int32( 4 ),
    ptErr_max = cms.double( 0.2 ),
    corner = cms.double( 1.479 ),
    PFClustersECAL = cms.InputTag( "particleFlowClusterECAL" ),
    eta_max = cms.double( 2.65 ),
    muonInputTag = cms.InputTag( "muons" ),
    eVetoDeltaCotTheta = cms.double( 100.0 ),
    maxd0cut = cms.double( 0.3 ),
    PFClustersHFEM = cms.InputTag( "particleFlowClusterHFEM" ),
    d0cutb = cms.double( 0.5 ),
    checkTrackPropagation = cms.bool( False ),
    usePFClusters = cms.bool( False ),
    vertexMaxDZ = cms.double( 0.2 ),
    deltaRShower = cms.double( 0.01 ),
    chi2_max = cms.double( 5.0 ),
    maxpt_eta25 = cms.double( 0.0 ),
    track_algos = cms.vint32(  ),
    ptErr_tight_max = cms.double( 0.2 ),
    maxTrackAlgo = cms.int32( 8 ),
    nLayers = cms.int32( 0 ),
    correctShowerTracks = cms.bool( False ),
    vetoDuplicates = cms.bool( False ),
    pt_max = cms.double( 100.0 ),
    radius = cms.double( 130.0 ),
    nhits_tight_min = cms.double( 9.0 ),
    beamSpotInputTag = cms.InputTag( "unused" ),
    dupDCotTh = cms.double( 6.0E-4 ),
    usedeltaRRejection = cms.bool( False ),
    trackInputTag = cms.InputTag( "generalTracks" ),
    dupDPhi = cms.double( 0.03 ),
    electronInputTag = cms.InputTag( "gsfElectrons" ),
    tcmetDepValueMap = cms.InputTag( 'muonTCMETValueMapProducer','muCorrData' ),
    d0cuta = cms.double( 0.015 ),
    hOverECut = cms.double( 0.1 ),
    electronVetoCone = cms.bool( True ),
    muonDepValueMap = cms.InputTag( 'muonMETValueMapProducer','muCorrData' ),
    metInputTag = cms.InputTag( "met" ),
    usePvtxd0 = cms.bool( False ),
    vertexInputTag = cms.InputTag( "offlinePrimaryVertices" ),
    zdist = cms.double( 314.0 ),
    nhits_min = cms.double( 6.0 ),
    eVetoDeltaR = cms.double( 0.015 ),
    maxpt_eta20 = cms.double( 100.0 ),
    pt_min = cms.double( 1.0 ),
    rf_type = cms.int32( 0 ),
    nMinOuterHits = cms.int32( 2 ),
    track_quality = cms.vint32( 2 ),
    isCosmics = cms.bool( False ),
    eVetoMinElectronPt = cms.double( 10.0 )
)
process.hltGlobalSumETHfFilter100 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 100.0 ),
    Max = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPAMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hlt1PAVertexFilter = cms.EDFilter( "HLTVertexFilter",
    maxZ = cms.double( 20.0 ),
    saveTags = cms.bool( False ),
    minVertices = cms.uint32( 2 ),
    maxChi2 = cms.double( 99999.0 ),
    inputTag = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    minNDoF = cms.double( 3.0 ),
    maxD0 = cms.double( 0.5 )
)
process.hltPAPixelCandsForHFSumET = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPAPixelTracksForHighMult" ),
    particleType = cms.string( "pi+" )
)
process.hlt1PAHighMult3ForHFSumET = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( False ),
    MinTrks = cms.int32( 3 ),
    MinPt = cms.double( 0.2 ),
    MaxVz = cms.double( 20.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltPAPixelCandsForHFSumET" ),
    vertexCollection = cms.InputTag( "hltPAPixelVerticesForHighMult" ),
    MaxPt = cms.double( 9999.0 ),
    MinSep = cms.double( 0.4 )
)
process.hltPrePAHFSumET100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAHFSumET140 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilter140 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 140.0 ),
    Max = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPAMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltPrePAHFSumET170 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilter170 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 170.0 ),
    Max = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPAMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltPrePAHFSumET210 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilter210 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 210.0 ),
    Max = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPAMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltL1sRomanPotsTech52 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "52" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPrePARomanPotsTech52 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1Tech53MB = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "53" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPrePAL1Tech53MB = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAL1Tech53MBSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAPixelTracksForMinBias = cms.EDProducer( "PixelTrackProducer",
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      ptMin = cms.double( 0.4 ),
      tipMax = cms.double( 1.0 )
    ),
    useFilterWithES = cms.bool( False ),
    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originHalfLength = cms.double( 25.0 ),
        originRadius = cms.double( 0.1 ),
        ptMin = cms.double( 0.4 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "hltESPPixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.06 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      )
    )
)
process.hltPAPixelCandsForMinBias = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPAPixelTracksForMinBias" ),
    particleType = cms.string( "pi+" )
)
process.hltPAMinBiasPixelFilter1 = cms.EDFilter( "HLTPixlMBFilt",
    saveTags = cms.bool( False ),
    pixlTag = cms.InputTag( "hltPAPixelCandsForMinBias" ),
    MinTrks = cms.uint32( 1 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 1.0 )
)
process.hltL1sL1Tech54ZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "54" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPrePAL1Tech54ZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sT1minbiasTech55 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "55" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPrePAT1minbiasTech55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sTechTrigHCALNoise = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "11 OR 12" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPrePAL1TechHBHEHOtotalOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1Tech63CASTORHaloMuon = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "63" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPrePAL1Tech63CASTORHaloMuon = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1CastorEmTotemLowMultiplicity = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_CastorEm_TotemLowMultiplicity" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePACastorEmTotemLowMultiplicity = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPixelTrackMultVetoFilterCastor = cms.EDFilter( "HLTPixlMBFilt",
    saveTags = cms.bool( False ),
    pixlTag = cms.InputTag( "hltPAPixelCandsForMinBias" ),
    MinTrks = cms.uint32( 10 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 0.0 )
)
process.hltL1sL1CastorEmNotHfCoincidencePm = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_CastorEm_NotHcalHfCoincidencePm" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePACastorEmNotHfCoincidencePm = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPixelTrackFilterCastorHfMin = cms.EDFilter( "HLTPixlMBFilt",
    saveTags = cms.bool( False ),
    pixlTag = cms.InputTag( "hltPAPixelCandsForMinBias" ),
    MinTrks = cms.uint32( 1 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 0.0 )
)
process.hltPixelTrackMultVetoFilterCastorHfMax = cms.EDFilter( "HLTPixlMBFilt",
    saveTags = cms.bool( False ),
    pixlTag = cms.InputTag( "hltPAPixelCandsForMinBias" ),
    MinTrks = cms.uint32( 3 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 0.0 )
)
process.hltL1sL1CastorEmNotHfSingleChannel = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_CastorEm_NotHcalHfSingleChannel" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePACastorEmNotHfSingleChannel = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1CastorTotalTotemLowMultiplicity = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_CastorTotalEnergy_TotemLowMultiplicity" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1CastorTotalTotemLowMultiplicity = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sPAMinBiasHFBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincidencePm_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMinBiasHF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sPAMinBiasHFORBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfSingleChannel_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMinBiasHFOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sBscMinBiasThreshold1BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasThreshold1_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMinBiasBHC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sPAMinBiasBscBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasOR_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMinBiasBHCOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sPAMinBiasHfBptxANDorBscBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincidencePm_BptxAND OR L1_BscMinBiasThreshold1_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMinBiasHfOrBHC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1BptxPlusNotBptxMinus = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxPlus_NotBptxMinus" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePABptxPlusNotBptxMinus = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1BptxMinusNotBptxPlus = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxMinus_NotBptxPlus" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePABptxMinusNotBptxPlus = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAZeroBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAHFORSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAZeroBiasPixelDoubleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAMinBiasPixelFilter2 = cms.EDFilter( "HLTPixlMBFilt",
    saveTags = cms.bool( False ),
    pixlTag = cms.InputTag( "hltPAPixelCandsForMinBias" ),
    MinTrks = cms.uint32( 2 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 1.0 )
)
process.hltPrePADoubleMu4Acoplanarity03 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL2fL1sL1DoubleMu0L1f0L2f0 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( False ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sL1DoubleMu0L1f0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 3.0 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL3fL1sL1DoubleMu0L1f0L2f0L3f4 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL2fL1sL1DoubleMu0L1f0L2f0" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 2 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.15 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 4.0 )
)
process.hltDoubleMu4ExclL3PreFiltered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( False ),
    ChargeOpt = cms.int32( -1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltL2fL1sL1DoubleMu0L1f0L2f0" ),
    MaxPtBalance = cms.double( 3.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 0.3 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 9999.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 2.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 99999.9 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.15 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 0.0 )
)
process.hltPrePAExclDijet35HFOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltExclDiJet35HFOR = cms.EDFilter( "HLTExclDiCaloJetFilter",
    saveTags = cms.bool( True ),
    inputJetTag = cms.InputTag( "hltCaloJetCorrected" ),
    triggerType = cms.int32( 85 ),
    minPtJet = cms.double( 35.0 ),
    HF_OR = cms.bool( True ),
    minHFe = cms.double( 50.0 )
)
process.hltL1sL1SingleJet16FwdVeto5 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_FwdVeto5" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAExclDijet35HFAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltExclDiJet35HFAND = cms.EDFilter( "HLTExclDiCaloJetFilter",
    saveTags = cms.bool( False ),
    inputJetTag = cms.InputTag( "hltCaloJetCorrected" ),
    triggerType = cms.int32( 85 ),
    minPtJet = cms.double( 35.0 ),
    HF_OR = cms.bool( False ),
    minHFe = cms.double( 50.0 )
)
process.hltL1sL1DoubleEG3FwdVeto = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG3_FwdVeto" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1DoubleEG3FwdVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet52TotemDiffractive = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet52_TotemDiffractive" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1SingleJet52TotemDiffractive = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleMu20TotemDiffractive = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu20_TotemDiffractive" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1SingleMu20TotemDiffractive = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleEG20TotemDiffractive = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG20_TotemDiffractive" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1SingleEG20TotemDiffractive = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleJet20TotemDiffractive = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleJet20_TotemDiffractive" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1DoubleJet20TotemDiffractive = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleJetC36TotemDiffractive = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleJetC36_TotemDiffractive" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1DoubleJetC36TotemDiffractive = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleMu5TotemDiffractive = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu5_TotemDiffractive" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1DoubleMu5TotemDiffractive = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleEG5TotemDiffractive = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5_TotemDiffractive" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAL1DoubleEG5TotemDiffractive = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleForJet16EtaOpp = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleForJet16_EtaOpp" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePADoubleJet20ForwardBackward = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltDoubleJet20ForwardBackward = cms.EDFilter( "HLTForwardBackwardCaloJetsFilter",
    saveTags = cms.bool( False ),
    minPt = cms.double( 20.0 ),
    maxEta = cms.double( 5.1 ),
    minEta = cms.double( 3.0 ),
    inputTag = cms.InputTag( "hltCaloJetCorrected" ),
    nTot = cms.uint32( 0 ),
    nPos = cms.uint32( 1 ),
    triggerType = cms.int32( 85 ),
    nNeg = cms.uint32( 1 )
)
process.hltL1sL1Mu0EG5 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_EG5" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAMu7Ele7CaloIdTCaloIsoVL = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1Mu0EG5L1MuFiltered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1Mu0EG5" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1Mu0EG5L2MuFiltered0 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltL1Mu0EG5L1MuFiltered0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1Mu0EG5L3MuFiltered7 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1Mu0EG5L2MuFiltered0" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 7.0 )
)
process.hltEGRegionalL1Mu0EG5 = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    saveTags = cms.bool( False ),
    endcap_end = cms.double( 2.65 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    barrel_end = cms.double( 1.4791 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candIsolatedTag = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    region_phi_size = cms.double( 1.044 ),
    region_eta_size = cms.double( 0.522 ),
    L1SeedFilterTag = cms.InputTag( "hltL1sL1Mu0EG5" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    candNonIsolatedTag = cms.InputTag( "" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' )
)
process.hltEG7EtFilterL1Mu0EG5 = cms.EDFilter( "HLTEgammaEtFilter",
    saveTags = cms.bool( False ),
    L1NonIsoCand = cms.InputTag( "" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    inputTag = cms.InputTag( "hltEGRegionalL1Mu0EG5" ),
    etcutEB = cms.double( 7.0 ),
    ncandcut = cms.int32( 1 ),
    etcutEE = cms.double( 7.0 )
)
process.hltMu7Ele7CaloIdTCaloIsoVLClusterShapeFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.031 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.011 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededHLTClusterShape" ),
    candTag = cms.InputTag( "hltEG7EtFilterL1Mu0EG5" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltMu7Ele7CaloIdTCaloIsoVLTHEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.1 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.15 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalForHE" ),
    candTag = cms.InputTag( "hltMu7Ele7CaloIdTCaloIsoVLClusterShapeFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltMu7Ele7CaloIdTCaloIsoVLEcalIsoFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.2 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.2 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonEcalIso" ),
    candTag = cms.InputTag( "hltMu7Ele7CaloIdTCaloIsoVLTHEFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltMu7Ele7CaloIdTCaloIsoVLHcalIsoFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    nonIsoTag = cms.InputTag( "" ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.2 ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    thrOverEEB = cms.double( 0.2 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( True ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltL1SeededPhotonHcalIso" ),
    candTag = cms.InputTag( "hltMu7Ele7CaloIdTCaloIsoVLEcalIsoFilter" ),
    thrOverE2EE = cms.double( -1.0 )
)
process.hltMu7Ele7CaloIdTPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    saveTags = cms.bool( True ),
    doIsolated = cms.bool( True ),
    L1NonIsoCand = cms.InputTag( "" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "" ),
    L1IsoCand = cms.InputTag( "hltL1SeededRecoEcalCandidate" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltMu7Ele7CaloIdTCaloIsoVLHcalIsoFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1SeededStartUpElectronPixelSeeds" )
)
process.hltL1sPASingleEG5BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPrePAUpcSingleEG5PixelTrackVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPACountPAPixFilter10 = cms.EDFilter( "CandViewCountFilter",
    src = cms.InputTag( "hltPAPixelCandsForMinBias" ),
    minNumber = cms.uint32( 10 )
)
process.hltPrePAUpcSingleEG5FullTrackVeto7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAUpcTrackSeeds = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    InputVertexCollection = cms.InputTag( "" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltPAPixelTracksForMinBias" ),
    originRadius = cms.double( 1.0E9 )
)
process.hltPAUpcCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltPAUpcTrackSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPMuTrackJpsiTrajectoryBuilder" )
)
process.hltPAUpcCtfTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltPAUpcCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherRK" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "hltMuTrackJpsiCtfTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltPAUpcCtfTrackCands = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPAUpcCtfTracks" ),
    particleType = cms.string( "mu-" )
)
process.hltPACountUpcTrackFilter1 = cms.EDFilter( "CandViewCountFilter",
    src = cms.InputTag( "hltPAUpcCtfTrackCands" ),
    minNumber = cms.uint32( 1 )
)
process.hltPACountUpcTrackFilter7 = cms.EDFilter( "CandViewCountFilter",
    src = cms.InputTag( "hltPAUpcCtfTrackCands" ),
    minNumber = cms.uint32( 7 )
)
process.hltPrePAUpcSingleMuOpenPixelTrackVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAUpcSingleMuOpenFullTrackVeto7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPrePAUpcSingleMuOpenTkMuOnia = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPAUpcSingleMuOpenTkMuOniaDCAL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMuOpen" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPAUpcSingleMuOpenTkMuOniaDCAL2Filtered1 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltPAUpcSingleMuOpenTkMuOniaDCAL1Filtered0" ),
    MinPt = cms.double( 1.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPAUpcSingleMuOpenTkMuOniaDCAL3Filtered1 = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltPAUpcSingleMuOpenTkMuOniaDCAL2Filtered1" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 1.0 )
)
process.hltMuTrackPixelTrackSelectorUpcOnia = cms.EDProducer( "QuarkoniaTrackSelector",
    MinTrackPt = cms.double( 1.0 ),
    muonCandidates = cms.InputTag( "hltL3MuonCandidates" ),
    MaxTrackEta = cms.double( 999.0 ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    MaxMasses = cms.vdouble( 1.0E20 ),
    checkCharge = cms.bool( False ),
    MinMasses = cms.vdouble( 0.0 ),
    MinTrackP = cms.double( 1.0 )
)
process.hltMuTrackPixelTrackCandsUpcOnia = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltMuTrackPixelTrackSelectorUpcOnia" ),
    particleType = cms.string( "mu-" )
)
process.hltMuOpenTrack1PixelMassFilteredUpcOnia = cms.EDFilter( "HLTMuonTrackMassFilter",
    saveTags = cms.bool( False ),
    MaxDCAMuonTrack = cms.double( 99999.9 ),
    PreviousCandTag = cms.InputTag( "hltPAUpcSingleMuOpenTkMuOniaDCAL3Filtered1" ),
    TrackTag = cms.InputTag( "hltMuTrackPixelTrackCandsUpcOnia" ),
    MaxTrackDz = cms.double( 999.0 ),
    MaxTrackNormChi2 = cms.double( 1.0E10 ),
    MinTrackPt = cms.double( 0.0 ),
    MinTrackHits = cms.int32( 3 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MaxMasses = cms.vdouble( 1.0E20 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxTrackEta = cms.double( 999.0 ),
    checkCharge = cms.bool( False ),
    MinMasses = cms.vdouble( 0.0 ),
    CutCowboys = cms.bool( False ),
    MaxTrackDxy = cms.double( 999.0 ),
    MinTrackP = cms.double( 0.0 )
)
process.hltMuTrackTrackSeedsUpcOnia = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    InputVertexCollection = cms.InputTag( "" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltMuTrackPixelTrackSelectorUpcOnia" ),
    originRadius = cms.double( 1.0E9 )
)
process.hltMuTrackCkfTrackCandidatesUpcOnia = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltMuTrackTrackSeedsUpcOnia" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPMuTrackJpsiTrajectoryBuilder" )
)
process.hltMuTrackCtfTracksUpcOnia = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltMuTrackCkfTrackCandidatesUpcOnia" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPFittingSmootherRK" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "hltMuTrackJpsiCtfTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltMuTrackCtfTrackCandsUpcOnia = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltMuTrackCtfTracksUpcOnia" ),
    particleType = cms.string( "mu-" )
)
process.hltMuOpenTkMu1TrackMassFilteredUpcOnia = cms.EDFilter( "HLTMuonTrackMassFilter",
    saveTags = cms.bool( True ),
    MaxDCAMuonTrack = cms.double( 99999.9 ),
    PreviousCandTag = cms.InputTag( "hltMuOpenTrack1PixelMassFilteredUpcOnia" ),
    TrackTag = cms.InputTag( "hltMuTrackCtfTrackCandsUpcOnia" ),
    MaxTrackDz = cms.double( 999.0 ),
    MaxTrackNormChi2 = cms.double( 1.0E10 ),
    MinTrackPt = cms.double( 1.0 ),
    MinTrackHits = cms.int32( 5 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MaxMasses = cms.vdouble( 1.0E20 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxTrackEta = cms.double( 999.0 ),
    checkCharge = cms.bool( True ),
    MinMasses = cms.vdouble( 0.0 ),
    CutCowboys = cms.bool( False ),
    MaxTrackDxy = cms.double( 999.0 ),
    MinTrackP = cms.double( 1.0 )
)
process.hltMuTkMuMuonLinksUpcOnia = cms.EDProducer( "MuonLinksProducerForHLT",
    pMin = cms.double( 1.0 ),
    InclusiveTrackerTrackCollection = cms.InputTag( "hltMuTrackCtfTracksUpcOnia" ),
    shareHitFraction = cms.double( 0.8 ),
    LinkCollection = cms.InputTag( "hltL3MuonsLinksCombination" ),
    ptMin = cms.double( 1.0 )
)
process.hltMuTkMuMuonsUpcOnia = cms.EDProducer( "MuonIdProducer",
    TrackExtractorPSet = cms.PSet( 
      Diff_z = cms.double( 0.2 ),
      inputTrackCollection = cms.InputTag( "hltPFMuonMerging" ),
      BeamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
      ComponentName = cms.string( "TrackExtractor" ),
      DR_Max = cms.double( 1.0 ),
      Diff_r = cms.double( 0.1 ),
      Chi2Prob_Min = cms.double( -1.0 ),
      DR_Veto = cms.double( 0.01 ),
      NHits_Min = cms.uint32( 0 ),
      Chi2Ndof_Max = cms.double( 1.0E64 ),
      Pt_Min = cms.double( -1.0 ),
      DepositLabel = cms.untracked.string( "" ),
      BeamlineOption = cms.string( "BeamSpotFromEvent" )
    ),
    maxAbsEta = cms.double( 3.0 ),
    fillGlobalTrackRefits = cms.bool( False ),
    arbitrationCleanerOptions = cms.PSet( 
      Clustering = cms.bool( True ),
      ME1a = cms.bool( True ),
      ClusterDPhi = cms.double( 0.6 ),
      OverlapDTheta = cms.double( 0.02 ),
      Overlap = cms.bool( True ),
      OverlapDPhi = cms.double( 0.0786 ),
      ClusterDTheta = cms.double( 0.02 )
    ),
    globalTrackQualityInputTag = cms.InputTag( "glbTrackQual" ),
    addExtraSoftMuons = cms.bool( False ),
    debugWithTruthMatching = cms.bool( False ),
    CaloExtractorPSet = cms.PSet( 
      PrintTimeReport = cms.untracked.bool( False ),
      DR_Max = cms.double( 1.0 ),
      DepositInstanceLabels = cms.vstring( 'ecal',
        'hcal',
        'ho' ),
      Noise_HE = cms.double( 0.2 ),
      NoiseTow_EB = cms.double( 0.04 ),
      NoiseTow_EE = cms.double( 0.15 ),
      Threshold_H = cms.double( 0.5 ),
      ServiceParameters = cms.PSet( 
        Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False )
      ),
      Threshold_E = cms.double( 0.2 ),
      PropagatorName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      DepositLabel = cms.untracked.string( "Cal" ),
      UseRecHitsFlag = cms.bool( False ),
      TrackAssociatorParameters = cms.PSet( 
        muonMaxDistanceSigmaX = cms.double( 0.0 ),
        muonMaxDistanceSigmaY = cms.double( 0.0 ),
        CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
        dRHcal = cms.double( 1.0 ),
        dRPreshowerPreselection = cms.double( 0.2 ),
        CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForPF" ),
        useEcal = cms.bool( False ),
        dREcal = cms.double( 1.0 ),
        dREcalPreselection = cms.double( 1.0 ),
        HORecHitCollectionLabel = cms.InputTag( "hltHoreco" ),
        dRMuon = cms.double( 9999.0 ),
        propagateAllDirections = cms.bool( True ),
        muonMaxDistanceX = cms.double( 5.0 ),
        muonMaxDistanceY = cms.double( 5.0 ),
        useHO = cms.bool( False ),
        trajectoryUncertaintyTolerance = cms.double( -1.0 ),
        usePreshower = cms.bool( False ),
        DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
        EERecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
        dRHcalPreselection = cms.double( 1.0 ),
        useMuon = cms.bool( False ),
        useCalo = cms.bool( True ),
        accountForTrajectoryChangeCalo = cms.bool( False ),
        EBRecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
        dRMuonPreselection = cms.double( 0.2 ),
        truthMatch = cms.bool( False ),
        HBHERecHitCollectionLabel = cms.InputTag( "hltHbhereco" ),
        useHcal = cms.bool( False )
      ),
      Threshold_HO = cms.double( 0.5 ),
      Noise_EE = cms.double( 0.1 ),
      Noise_EB = cms.double( 0.025 ),
      DR_Veto_H = cms.double( 0.1 ),
      CenterConeOnCalIntersection = cms.bool( False ),
      ComponentName = cms.string( "CaloExtractorByAssociator" ),
      Noise_HB = cms.double( 0.2 ),
      DR_Veto_E = cms.double( 0.07 ),
      DR_Veto_HO = cms.double( 0.1 ),
      Noise_HO = cms.double( 0.2 )
    ),
    runArbitrationCleaner = cms.bool( False ),
    fillEnergy = cms.bool( False ),
    TrackerKinkFinderParameters = cms.PSet( 
      usePosition = cms.bool( False ),
      diagonalOnly = cms.bool( False )
    ),
    TimingFillerParameters = cms.PSet( 
      UseDT = cms.bool( True ),
      ErrorDT = cms.double( 6.0 ),
      EcalEnergyCut = cms.double( 0.4 ),
      ErrorEB = cms.double( 2.085 ),
      ErrorCSC = cms.double( 7.4 ),
      CSCTimingParameters = cms.PSet( 
        CSCsegments = cms.InputTag( "hltCscSegments" ),
        CSCTimeOffset = cms.double( 0.0 ),
        CSCStripTimeOffset = cms.double( 0.0 ),
        MatchParameters = cms.PSet( 
          CSCsegments = cms.InputTag( "hltCscSegments" ),
          DTsegments = cms.InputTag( "hltDt4DSegments" ),
          DTradius = cms.double( 0.01 ),
          TightMatchDT = cms.bool( False ),
          TightMatchCSC = cms.bool( True )
        ),
        debug = cms.bool( False ),
        UseStripTime = cms.bool( True ),
        CSCStripError = cms.double( 7.0 ),
        CSCWireError = cms.double( 8.6 ),
        CSCWireTimeOffset = cms.double( 0.0 ),
        ServiceParameters = cms.PSet( 
          Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
          RPCLayers = cms.bool( True )
        ),
        PruneCut = cms.double( 100.0 ),
        UseWireTime = cms.bool( True )
      ),
      DTTimingParameters = cms.PSet( 
        HitError = cms.double( 6.0 ),
        DoWireCorr = cms.bool( False ),
        MatchParameters = cms.PSet( 
          CSCsegments = cms.InputTag( "hltCscSegments" ),
          DTsegments = cms.InputTag( "hltDt4DSegments" ),
          DTradius = cms.double( 0.01 ),
          TightMatchDT = cms.bool( False ),
          TightMatchCSC = cms.bool( True )
        ),
        debug = cms.bool( False ),
        DTsegments = cms.InputTag( "hltDt4DSegments" ),
        PruneCut = cms.double( 10000.0 ),
        RequireBothProjections = cms.bool( False ),
        HitsMin = cms.int32( 5 ),
        DTTimeOffset = cms.double( 2.7 ),
        DropTheta = cms.bool( True ),
        UseSegmentT0 = cms.bool( False ),
        ServiceParameters = cms.PSet( 
          Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
          RPCLayers = cms.bool( True )
        )
      ),
      ErrorEE = cms.double( 6.95 ),
      UseCSC = cms.bool( True ),
      UseECAL = cms.bool( True )
    ),
    inputCollectionTypes = cms.vstring( 'inner tracks',
      'links' ),
    minCaloCompatibility = cms.double( 0.6 ),
    ecalDepositName = cms.string( "ecal" ),
    minP = cms.double( 0.0 ),
    fillIsolation = cms.bool( False ),
    jetDepositName = cms.string( "jets" ),
    hoDepositName = cms.string( "ho" ),
    writeIsoDeposits = cms.bool( False ),
    maxAbsPullX = cms.double( 4.0 ),
    maxAbsPullY = cms.double( 9999.0 ),
    minPt = cms.double( 1.0 ),
    TrackAssociatorParameters = cms.PSet( 
      muonMaxDistanceSigmaX = cms.double( 0.0 ),
      muonMaxDistanceSigmaY = cms.double( 0.0 ),
      CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
      dRHcal = cms.double( 9999.0 ),
      dRPreshowerPreselection = cms.double( 0.2 ),
      CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForPF" ),
      useEcal = cms.bool( False ),
      dREcal = cms.double( 9999.0 ),
      dREcalPreselection = cms.double( 0.05 ),
      HORecHitCollectionLabel = cms.InputTag( "hltHoreco" ),
      dRMuon = cms.double( 9999.0 ),
      propagateAllDirections = cms.bool( True ),
      muonMaxDistanceX = cms.double( 5.0 ),
      muonMaxDistanceY = cms.double( 5.0 ),
      useHO = cms.bool( False ),
      trajectoryUncertaintyTolerance = cms.double( -1.0 ),
      usePreshower = cms.bool( False ),
      DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
      EERecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
      dRHcalPreselection = cms.double( 0.2 ),
      useMuon = cms.bool( True ),
      useCalo = cms.bool( False ),
      accountForTrajectoryChangeCalo = cms.bool( False ),
      EBRecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
      dRMuonPreselection = cms.double( 0.2 ),
      truthMatch = cms.bool( False ),
      HBHERecHitCollectionLabel = cms.InputTag( "hltHbhereco" ),
      useHcal = cms.bool( False )
    ),
    JetExtractorPSet = cms.PSet( 
      PrintTimeReport = cms.untracked.bool( False ),
      ExcludeMuonVeto = cms.bool( True ),
      TrackAssociatorParameters = cms.PSet( 
        muonMaxDistanceSigmaX = cms.double( 0.0 ),
        muonMaxDistanceSigmaY = cms.double( 0.0 ),
        CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
        dRHcal = cms.double( 0.5 ),
        dRPreshowerPreselection = cms.double( 0.2 ),
        CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForPF" ),
        useEcal = cms.bool( False ),
        dREcal = cms.double( 0.5 ),
        dREcalPreselection = cms.double( 0.5 ),
        HORecHitCollectionLabel = cms.InputTag( "hltHoreco" ),
        dRMuon = cms.double( 9999.0 ),
        propagateAllDirections = cms.bool( True ),
        muonMaxDistanceX = cms.double( 5.0 ),
        muonMaxDistanceY = cms.double( 5.0 ),
        useHO = cms.bool( False ),
        trajectoryUncertaintyTolerance = cms.double( -1.0 ),
        usePreshower = cms.bool( False ),
        DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
        EERecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
        dRHcalPreselection = cms.double( 0.5 ),
        useMuon = cms.bool( False ),
        useCalo = cms.bool( True ),
        accountForTrajectoryChangeCalo = cms.bool( False ),
        EBRecHitCollectionLabel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
        dRMuonPreselection = cms.double( 0.2 ),
        truthMatch = cms.bool( False ),
        HBHERecHitCollectionLabel = cms.InputTag( "hltHbhereco" ),
        useHcal = cms.bool( False )
      ),
      ServiceParameters = cms.PSet( 
        Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' ),
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False )
      ),
      ComponentName = cms.string( "JetExtractor" ),
      DR_Max = cms.double( 1.0 ),
      PropagatorName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      JetCollectionLabel = cms.InputTag( "hltAntiKT5CaloJetsPFEt5" ),
      DR_Veto = cms.double( 0.1 ),
      Threshold = cms.double( 5.0 )
    ),
    fillGlobalTrackQuality = cms.bool( False ),
    minPCaloMuon = cms.double( 1.0E9 ),
    maxAbsDy = cms.double( 9999.0 ),
    fillCaloCompatibility = cms.bool( False ),
    fillMatching = cms.bool( True ),
    MuonCaloCompatibility = cms.PSet( 
      allSiPMHO = cms.bool( False ),
      PionTemplateFileName = cms.FileInPath( "RecoMuon/MuonIdentification/data/MuID_templates_pions_lowPt_3_1_norm.root" ),
      MuonTemplateFileName = cms.FileInPath( "RecoMuon/MuonIdentification/data/MuID_templates_muons_lowPt_3_1_norm.root" ),
      delta_eta = cms.double( 0.02 ),
      delta_phi = cms.double( 0.02 )
    ),
    fillTrackerKink = cms.bool( False ),
    hcalDepositName = cms.string( "hcal" ),
    sigmaThresholdToFillCandidateP4WithGlobalFit = cms.double( 2.0 ),
    inputCollectionLabels = cms.VInputTag( 'hltMuTrackCtfTracksUpcOnia','hltMuTkMuMuonLinksUpcOnia' ),
    trackDepositName = cms.string( "tracker" ),
    maxAbsDx = cms.double( 3.0 ),
    ptThresholdToFillCandidateP4WithGlobalFit = cms.double( 200.0 ),
    minNumberOfMatches = cms.int32( 1 )
)
process.hltMuTkMuTrackerMuonCandsUpcOnia = cms.EDProducer( "L3MuonCandidateProducerFromMuons",
    InputObjects = cms.InputTag( "hltMuTkMuMuonsUpcOnia" )
)
process.hltMuOpenTkMu1TkMuMassFilteredUpcOnia = cms.EDFilter( "HLTMuonTrackMassFilter",
    saveTags = cms.bool( True ),
    MaxDCAMuonTrack = cms.double( 0.5 ),
    PreviousCandTag = cms.InputTag( "hltMuOpenTkMu1TrackMassFilteredUpcOnia" ),
    TrackTag = cms.InputTag( "hltMuTkMuTrackerMuonCandsUpcOnia" ),
    MaxTrackDz = cms.double( 999.0 ),
    MaxTrackNormChi2 = cms.double( 1.0E10 ),
    MinTrackPt = cms.double( 1.0 ),
    MinTrackHits = cms.int32( 5 ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    MaxMasses = cms.vdouble( 12.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxTrackEta = cms.double( 999.0 ),
    checkCharge = cms.bool( True ),
    MinMasses = cms.vdouble( 2.0 ),
    CutCowboys = cms.bool( False ),
    MaxTrackDxy = cms.double( 999.0 ),
    MinTrackP = cms.double( 1.0 )
)
process.hltPrePARandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDQMFEDIntegrity = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltCSCMonitorModule = cms.EDAnalyzer( "CSCMonitorModule",
    BOOKING_XML_FILE = cms.FileInPath( "DQM/CSCMonitorModule/data/emuDQMBooking.xml" ),
    EventProcessor = cms.untracked.PSet( 
      PROCESS_EFF_PARAMETERS = cms.untracked.bool( False ),
      FRAEFF_AUTO_UPDATE = cms.untracked.bool( False ),
      EFF_NODATA_THRESHOLD = cms.untracked.double( 0.1 ),
      FRAEFF_AUTO_UPDATE_START = cms.untracked.uint32( 5 ),
      BINCHECK_MASK = cms.untracked.uint32( 384563190 ),
      BINCHECKER_CRC_CLCT = cms.untracked.bool( True ),
      EFF_COLD_SIGFAIL = cms.untracked.double( 5.0 ),
      PROCESS_DDU = cms.untracked.bool( False ),
      EFF_NODATA_SIGFAIL = cms.untracked.double( 5.0 ),
      BINCHECKER_MODE_DDU = cms.untracked.bool( False ),
      BINCHECKER_CRC_ALCT = cms.untracked.bool( True ),
      EFF_HOT_THRESHOLD = cms.untracked.double( 0.1 ),
      FOLDER_DDU = cms.untracked.string( "" ),
      BINCHECKER_CRC_CFEB = cms.untracked.bool( True ),
      EVENTS_ECHO = cms.untracked.uint32( 1000 ),
      DDU_CHECK_MASK = cms.untracked.uint32( 4294959103 ),
      FRAEFF_SEPARATE_THREAD = cms.untracked.bool( False ),
      EFF_HOT_SIGFAIL = cms.untracked.double( 5.0 ),
      FOLDER_PAR = cms.untracked.string( "" ),
      FRAEFF_AUTO_UPDATE_FREQ = cms.untracked.uint32( 200 ),
      EFF_COLD_THRESHOLD = cms.untracked.double( 0.1 ),
      FOLDER_EMU = cms.untracked.string( "CSC/FEDIntegrity_EvF" ),
      DDU_BINCHECK_MASK = cms.untracked.uint32( 384563190 ),
      EFF_ERR_SIGFAIL = cms.untracked.double( 5.0 ),
      PROCESS_CSC = cms.untracked.bool( False ),
      PROCESS_EFF_HISTOS = cms.untracked.bool( False ),
      MO_FILTER = cms.untracked.vstring( '-/^.*$/',
        '+/FEDEntries/',
        '+/FEDFatal/',
        '+/FEDFormatFatal/',
        '+/FEDNonFatal/',
        '+/^CSC_Reporting$/',
        '+/^CSC_Format_Errors$/',
        '+/^CSC_Format_Warnings$/',
        '+/^CSC_L1A_out_of_sync$/',
        '+/^CSC_wo_ALCT$/',
        '+/^CSC_wo_CFEB$/',
        '+/^CSC_wo_CLCT$/' ),
      FOLDER_CSC = cms.untracked.string( "" ),
      EFF_ERR_THRESHOLD = cms.untracked.double( 0.1 ),
      BINCHECKER_OUTPUT = cms.untracked.bool( False )
    ),
    InputObjects = cms.untracked.InputTag( "rawDataCollector" )
)
process.hltDTDQMEvF = cms.EDProducer( "DTUnpackingModule",
    useStandardFEDid = cms.bool( True ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    dataType = cms.string( "DDU" ),
    fedbyType = cms.bool( False ),
    readOutParameters = cms.PSet( 
      debug = cms.untracked.bool( False ),
      rosParameters = cms.PSet( 
        writeSC = cms.untracked.bool( True ),
        readingDDU = cms.untracked.bool( True ),
        performDataIntegrityMonitor = cms.untracked.bool( True ),
        readDDUIDfromDDU = cms.untracked.bool( True ),
        debug = cms.untracked.bool( False ),
        localDAQ = cms.untracked.bool( False )
      ),
      localDAQ = cms.untracked.bool( False ),
      performDataIntegrityMonitor = cms.untracked.bool( True )
    ),
    dqmOnly = cms.bool( True )
)
process.hltEcalRawToRecHitByproductProducer = cms.EDProducer( "EcalRawToRecHitByproductProducer",
    workerName = cms.string( "" )
)
process.hltEBHltTask = cms.EDAnalyzer( "EBHltTask",
    mergeRuns = cms.untracked.bool( False ),
    FEDRawDataCollection = cms.InputTag( "rawDataCollector" ),
    EBDetIdCollection3 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityGainSwitchErrors' ),
    EBDetIdCollection2 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityChIdErrors' ),
    EBDetIdCollection1 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityGainErrors' ),
    EBDetIdCollection0 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityDCCSizeErrors' ),
    enableCleanup = cms.untracked.bool( False ),
    folderName = cms.untracked.string( "FEDIntegrity_EvF" ),
    EcalElectronicsIdCollection3 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemTtIdErrors' ),
    EcalElectronicsIdCollection5 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemChIdErrors' ),
    EcalElectronicsIdCollection4 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemBlockSizeErrors' ),
    EcalElectronicsIdCollection6 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemGainErrors' ),
    EcalElectronicsIdCollection1 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityTTIdErrors' ),
    prefixME = cms.untracked.string( "EcalBarrel" ),
    EcalElectronicsIdCollection2 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityBlockSizeErrors' )
)
process.hltEEHltTask = cms.EDAnalyzer( "EEHltTask",
    mergeRuns = cms.untracked.bool( False ),
    FEDRawDataCollection = cms.InputTag( "rawDataCollector" ),
    enableCleanup = cms.untracked.bool( False ),
    folderName = cms.untracked.string( "FEDIntegrity_EvF" ),
    EEDetIdCollection0 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityDCCSizeErrors' ),
    EEDetIdCollection1 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityGainErrors' ),
    EEDetIdCollection2 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityChIdErrors' ),
    EEDetIdCollection3 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityGainSwitchErrors' ),
    EcalElectronicsIdCollection3 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemTtIdErrors' ),
    EcalElectronicsIdCollection5 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemChIdErrors' ),
    EcalElectronicsIdCollection4 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemBlockSizeErrors' ),
    EcalElectronicsIdCollection6 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityMemGainErrors' ),
    EcalElectronicsIdCollection1 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityTTIdErrors' ),
    prefixME = cms.untracked.string( "EcalEndcap" ),
    EcalElectronicsIdCollection2 = cms.InputTag( 'hltEcalRawToRecHitByproductProducer','EcalIntegrityBlockSizeErrors' )
)
process.hltESFEDIntegrityTask = cms.EDAnalyzer( "ESFEDIntegrityTask",
    FEDRawDataCollection = cms.InputTag( "rawDataCollector" ),
    ESDCCCollections = cms.InputTag( "NotUsed" ),
    ESKChipCollections = cms.InputTag( "NotUsed" ),
    FEDDirName = cms.untracked.string( "FEDIntegrity_EvF" ),
    prefixME = cms.untracked.string( "EcalPreshower" )
)
process.hltHcalDataIntegrityMonitor = cms.EDAnalyzer( "HcalDataIntegrityTask",
    mergeRuns = cms.untracked.bool( False ),
    UnpackerReportLabel = cms.untracked.InputTag( "hltHcalDigis" ),
    subSystemFolder = cms.untracked.string( "Hcal" ),
    skipOutOfOrderLS = cms.untracked.bool( False ),
    enableCleanup = cms.untracked.bool( False ),
    RawDataLabel = cms.untracked.InputTag( "rawDataCollector" ),
    NLumiBlocks = cms.untracked.int32( 4000 ),
    TaskFolder = cms.untracked.string( "FEDIntegrity_EvF" ),
    online = cms.untracked.bool( False ),
    debug = cms.untracked.int32( 0 ),
    AllowedCalibTypes = cms.untracked.vint32( 0, 1, 2, 3, 4, 5, 6, 7 )
)
process.hltL1tfed = cms.EDAnalyzer( "L1TFED",
    verbose = cms.untracked.bool( False ),
    DQMStore = cms.untracked.bool( True ),
    rawTag = cms.InputTag( "rawDataCollector" ),
    stableROConfig = cms.untracked.bool( True ),
    FEDDirName = cms.untracked.string( "L1T/FEDIntegrity_EvF" ),
    disableROOToutput = cms.untracked.bool( True ),
    L1FEDS = cms.vint32( 745, 760, 780, 812, 813 )
)
process.hltSiPixelHLTSource = cms.EDAnalyzer( "SiPixelHLTSource",
    saveFile = cms.untracked.bool( False ),
    outputFile = cms.string( "Pixel_DQM_HLT.root" ),
    slowDown = cms.untracked.bool( False ),
    ErrorInput = cms.InputTag( "hltSiPixelDigis" ),
    RawInput = cms.InputTag( "rawDataCollector" ),
    DirName = cms.untracked.string( "Pixel/FEDIntegrity_EvF" )
)
process.hltSiStripFEDCheck = cms.EDAnalyzer( "SiStripFEDCheckPlugin",
    PrintDebugMessages = cms.untracked.bool( False ),
    CheckChannelStatus = cms.untracked.bool( False ),
    DoPayloadChecks = cms.untracked.bool( False ),
    CheckChannelLengths = cms.untracked.bool( False ),
    WriteDQMStore = cms.untracked.bool( False ),
    CheckFELengths = cms.untracked.bool( False ),
    RawDataTag = cms.InputTag( "rawDataCollector" ),
    HistogramUpdateFrequency = cms.untracked.uint32( 1000 ),
    CheckChannelPacketCodes = cms.untracked.bool( False ),
    DirName = cms.untracked.string( "SiStrip/FEDIntegrity_EvF" )
)
process.hltRPCFEDIntegrity = cms.EDAnalyzer( "RPCFEDIntegrity",
    MaximumFEDID = cms.untracked.int32( 792 ),
    RPCRawCountsInputTag = cms.untracked.InputTag( "hltMuonRPCDigis" ),
    MinimumFEDID = cms.untracked.int32( 790 ),
    NumberOfFED = cms.untracked.int32( 3 ),
    RPCPrefixDir = cms.untracked.string( "RPC/FEDIntegrity_EvF" )
)
process.hltLogMonitorFilter = cms.EDFilter( "HLTLogMonitorFilter",
    saveTags = cms.bool( False ),
    default_threshold = cms.uint32( 10 ),
    categories = cms.VPSet( 
      cms.PSet(  name = cms.string( "TooManyTriplets" ),
        threshold = cms.uint32( 0 )
      ),
      cms.PSet(  name = cms.string( "Muon" ),
        threshold = cms.uint32( 0 )
      ),
      cms.PSet(  name = cms.string( "RecoMuon" ),
        threshold = cms.uint32( 0 )
      ),
      cms.PSet(  name = cms.string( "L3MuonCandidateProducer" ),
        threshold = cms.uint32( 0 )
      ),
      cms.PSet(  name = cms.string( "MatrixInversionFailure" ),
        threshold = cms.uint32( 0 )
      ),
      cms.PSet(  name = cms.string( "BasicTrajectoryState" ),
        threshold = cms.uint32( 0 )
      )
    )
)
process.hltPreLogMonitor = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023 )
)
process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
process.hltL1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    PrintVerbosity = cms.untracked.int32( 10 ),
    UseL1GlobalTriggerRecord = cms.bool( False ),
    PrintOutput = cms.untracked.int32( 3 ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
)
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.hltPreAOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCAP0Output = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCAPHISYMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCALUMIPIXELSOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDQMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreEcalCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreExpressOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreExpressOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( 'HLT_Mu15_eta2p1_v5',
      'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
      'HLT_PAL1SingleJet16_v1',
      'HLT_PAL1SingleJet36_v1',
      'HLT_PASingleForJet15_v1',
      'HLT_PASingleForJet25_v1',
      'HLT_PAJet20_NoJetID_v1',
      'HLT_PAJet40_NoJetID_v1',
      'HLT_PAJet60_NoJetID_v1',
      'HLT_PAJet80_NoJetID_v1',
      'HLT_PAJet100_NoJetID_v1',
      'HLT_PAJet120_NoJetID_v1',
      'HLT_PAForJet20Eta2_v1',
      'HLT_PAForJet40Eta2_v1',
      'HLT_PAForJet60Eta2_v1',
      'HLT_PAForJet80Eta2_v1',
      'HLT_PAForJet100Eta2_v1',
      'HLT_PAForJet20Eta3_v1',
      'HLT_PAForJet40Eta3_v1',
      'HLT_PAForJet60Eta3_v1',
      'HLT_PAForJet80Eta3_v1',
      'HLT_PAForJet100Eta3_v1',
      'HLT_PATripleJet20_20_20_v1',
      'HLT_PATripleJet40_20_20_v1',
      'HLT_PATripleJet60_20_20_v1',
      'HLT_PATripleJet80_20_20_v1',
      'HLT_PATripleJet100_20_20_v1',
      'HLT_PAJet40ETM30_v1',
      'HLT_PAJet60ETM30_v1',
      'HLT_PAL1DoubleMu0_v1',
      'HLT_PADimuon0_NoVertexing_v1',
      'HLT_PAL1DoubleMu0_HighQ_v1',
      'HLT_PAL1DoubleMuOpen_v1',
      'HLT_PAL2DoubleMu3_v1',
      'HLT_PAMu3_v1',
      'HLT_PAMu7_v1',
      'HLT_PAMu12_v1',
      'HLT_PABTagMu_Jet20_Mu4_v1',
      'HLT_PAMu3PFJet20_v1',
      'HLT_PAMu3PFJet40_v1',
      'HLT_PAMu7PFJet20_v1',
      'HLT_PAPhoton10_NoCaloIdVL_v1',
      'HLT_PAPhoton15_NoCaloIdVL_v1',
      'HLT_PAPhoton20_NoCaloIdVL_v1',
      'HLT_PAPhoton30_NoCaloIdVL_v1',
      'HLT_PAPhoton40_NoCaloIdVL_v1',
      'HLT_PAPhoton60_NoCaloIdVL_v1',
      'HLT_PAPhoton10_TightCaloIdVL_v1',
      'HLT_PAPhoton15_TightCaloIdVL_v1',
      'HLT_PAPhoton20_TightCaloIdVL_v1',
      'HLT_PAPhoton30_TightCaloIdVL_v1',
      'HLT_PAPhoton40_TightCaloIdVL_v1',
      'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
      'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
      'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
      'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
      'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
      'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
      'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
      'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
      'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
      'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
      'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
      'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
      'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
      'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
      'HLT_PAPixelTracks_Multiplicity100_v2',
      'HLT_PAPixelTracks_Multiplicity130_v2',
      'HLT_PAPixelTracks_Multiplicity160_v2',
      'HLT_PAPixelTracks_Multiplicity190_v2',
      'HLT_PAPixelTracks_Multiplicity220_v2',
      'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
      'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
      'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
      'HLT_PAFullTrack12_v2',
      'HLT_PAFullTrack20_v2',
      'HLT_PAFullTrack30_v2',
      'HLT_PAFullTrack50_v2',
      'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
      'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
      'HLT_PPPixelTracks_Multiplicity55_v1',
      'HLT_PPPixelTracks_Multiplicity70_v1',
      'HLT_PPPixelTracks_Multiplicity85_v1',
      'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
      'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
      'HLT_PPL1DoubleJetC36_v1',
      'HLT_PATech35_v1',
      'HLT_PATech35_HFSumET100_v3',
      'HLT_PAHFSumET100_v3',
      'HLT_PAHFSumET140_v3',
      'HLT_PAHFSumET170_v3',
      'HLT_PAHFSumET210_v3',
      'HLT_PARomanPots_Tech52_v1',
      'HLT_PAL1Tech53_MB_v1',
      'HLT_PAL1Tech53_MB_SingleTrack_v1',
      'HLT_PAL1Tech54_ZeroBias_v1',
      'HLT_PAT1minbias_Tech55_v1',
      'HLT_PAL1Tech_HBHEHO_totalOR_v1',
      'HLT_PAL1Tech63_CASTORHaloMuon_v1',
      'HLT_PACastorEmTotemLowMultiplicity_v1',
      'HLT_PACastorEmNotHfCoincidencePm_v1',
      'HLT_PACastorEmNotHfSingleChannel_v1',
      'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
      'HLT_PAMinBiasHF_v1',
      'HLT_PAMinBiasHF_OR_v1',
      'HLT_PAMinBiasBHC_v1',
      'HLT_PAMinBiasBHC_OR_v1',
      'HLT_PAMinBiasHfOrBHC_v1',
      'HLT_PABptxPlusNotBptxMinus_v1',
      'HLT_PABptxMinusNotBptxPlus_v1',
      'HLT_PAZeroBias_v1',
      'HLT_PAZeroBiasPixel_SingleTrack_v1',
      'HLT_PAHFOR_SingleTrack_v1',
      'HLT_PAZeroBiasPixel_DoubleTrack_v1',
      'HLT_PADoubleMu4_Acoplanarity03_v1',
      'HLT_PAExclDijet35_HFOR_v1',
      'HLT_PAExclDijet35_HFAND_v1',
      'HLT_PAL1DoubleEG3_FwdVeto_v1',
      'HLT_PAL1SingleJet52_TotemDiffractive_v1',
      'HLT_PAL1SingleMu20_TotemDiffractive_v1',
      'HLT_PAL1SingleEG20_TotemDiffractive_v1',
      'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
      'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
      'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
      'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
      'HLT_PADoubleJet20_ForwardBackward_v1',
      'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
      'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
      'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
      'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
      'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
      'HLT_PAUpcSingleMuOpenTkMu_Onia_v1' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreHLTDQMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHLTDQMOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( 'HLT_Mu15_eta2p1_v5',
      'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
      'HLT_PAHcalUTCA_v1',
      'HLT_PAHcalPhiSym_v1',
      'HLT_PAHcalNZS_v1',
      'HLT_PAL1SingleJet16_v1',
      'HLT_PAL1SingleJet36_v1',
      'HLT_PASingleForJet15_v1',
      'HLT_PASingleForJet25_v1',
      'HLT_PAJet20_NoJetID_v1',
      'HLT_PAJet40_NoJetID_v1',
      'HLT_PAJet60_NoJetID_v1',
      'HLT_PAJet80_NoJetID_v1',
      'HLT_PAJet100_NoJetID_v1',
      'HLT_PAJet120_NoJetID_v1',
      'HLT_PAForJet20Eta2_v1',
      'HLT_PAForJet40Eta2_v1',
      'HLT_PAForJet60Eta2_v1',
      'HLT_PAForJet80Eta2_v1',
      'HLT_PAForJet100Eta2_v1',
      'HLT_PAForJet20Eta3_v1',
      'HLT_PAForJet40Eta3_v1',
      'HLT_PAForJet60Eta3_v1',
      'HLT_PAForJet80Eta3_v1',
      'HLT_PAForJet100Eta3_v1',
      'HLT_PATripleJet20_20_20_v1',
      'HLT_PATripleJet40_20_20_v1',
      'HLT_PATripleJet60_20_20_v1',
      'HLT_PATripleJet80_20_20_v1',
      'HLT_PATripleJet100_20_20_v1',
      'HLT_PAJet40ETM30_v1',
      'HLT_PAJet60ETM30_v1',
      'HLT_PAL1DoubleMu0_v1',
      'HLT_PADimuon0_NoVertexing_v1',
      'HLT_PAL1DoubleMu0_HighQ_v1',
      'HLT_PAL1DoubleMuOpen_v1',
      'HLT_PAL2DoubleMu3_v1',
      'HLT_PAMu3_v1',
      'HLT_PAMu7_v1',
      'HLT_PAMu12_v1',
      'HLT_PABTagMu_Jet20_Mu4_v1',
      'HLT_PAMu3PFJet20_v1',
      'HLT_PAMu3PFJet40_v1',
      'HLT_PAMu7PFJet20_v1',
      'HLT_PAPhoton10_NoCaloIdVL_v1',
      'HLT_PAPhoton15_NoCaloIdVL_v1',
      'HLT_PAPhoton20_NoCaloIdVL_v1',
      'HLT_PAPhoton30_NoCaloIdVL_v1',
      'HLT_PAPhoton40_NoCaloIdVL_v1',
      'HLT_PAPhoton60_NoCaloIdVL_v1',
      'HLT_PAPhoton10_TightCaloIdVL_v1',
      'HLT_PAPhoton15_TightCaloIdVL_v1',
      'HLT_PAPhoton20_TightCaloIdVL_v1',
      'HLT_PAPhoton30_TightCaloIdVL_v1',
      'HLT_PAPhoton40_TightCaloIdVL_v1',
      'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
      'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
      'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
      'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
      'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
      'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
      'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
      'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
      'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
      'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
      'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
      'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
      'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
      'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
      'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
      'HLT_PAPixelTracks_Multiplicity100_v2',
      'HLT_PAPixelTracks_Multiplicity130_v2',
      'HLT_PAPixelTracks_Multiplicity160_v2',
      'HLT_PAPixelTracks_Multiplicity190_v2',
      'HLT_PAPixelTracks_Multiplicity220_v2',
      'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
      'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
      'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
      'HLT_PAFullTrack12_v2',
      'HLT_PAFullTrack20_v2',
      'HLT_PAFullTrack30_v2',
      'HLT_PAFullTrack50_v2',
      'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
      'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
      'HLT_PPPixelTracks_Multiplicity55_v1',
      'HLT_PPPixelTracks_Multiplicity70_v1',
      'HLT_PPPixelTracks_Multiplicity85_v1',
      'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
      'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
      'HLT_PPL1DoubleJetC36_v1',
      'HLT_PATech35_v1',
      'HLT_PATech35_HFSumET100_v3',
      'HLT_PAHFSumET100_v3',
      'HLT_PAHFSumET140_v3',
      'HLT_PAHFSumET170_v3',
      'HLT_PAHFSumET210_v3',
      'HLT_PARomanPots_Tech52_v1',
      'HLT_PAL1Tech53_MB_v1',
      'HLT_PAL1Tech53_MB_SingleTrack_v1',
      'HLT_PAL1Tech54_ZeroBias_v1',
      'HLT_PAT1minbias_Tech55_v1',
      'HLT_PAL1Tech_HBHEHO_totalOR_v1',
      'HLT_PAL1Tech63_CASTORHaloMuon_v1',
      'HLT_PACastorEmTotemLowMultiplicity_v1',
      'HLT_PACastorEmNotHfCoincidencePm_v1',
      'HLT_PACastorEmNotHfSingleChannel_v1',
      'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
      'HLT_PAMinBiasHF_v1',
      'HLT_PAMinBiasHF_OR_v1',
      'HLT_PAMinBiasBHC_v1',
      'HLT_PAMinBiasBHC_OR_v1',
      'HLT_PAMinBiasHfOrBHC_v1',
      'HLT_PABptxPlusNotBptxMinus_v1',
      'HLT_PABptxMinusNotBptxPlus_v1',
      'HLT_PAZeroBias_v1',
      'HLT_PAZeroBiasPixel_SingleTrack_v1',
      'HLT_PAHFOR_SingleTrack_v1',
      'HLT_PAZeroBiasPixel_DoubleTrack_v1',
      'HLT_PADoubleMu4_Acoplanarity03_v1',
      'HLT_PAExclDijet35_HFOR_v1',
      'HLT_PAExclDijet35_HFAND_v1',
      'HLT_PAL1DoubleEG3_FwdVeto_v1',
      'HLT_PAL1SingleJet52_TotemDiffractive_v1',
      'HLT_PAL1SingleMu20_TotemDiffractive_v1',
      'HLT_PAL1SingleEG20_TotemDiffractive_v1',
      'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
      'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
      'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
      'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
      'HLT_PADoubleJet20_ForwardBackward_v1',
      'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
      'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
      'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
      'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
      'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
      'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
      'HLT_PARandom_v1' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreNanoDSTOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreRPCMONOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreTrackerCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)

process.hltOutputA = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputA.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_Activity_Ecal_SC7_v13',
  'HLT_BeamGas_HF_Beam1_v5',
  'HLT_BeamGas_HF_Beam2_v5',
  'HLT_BeamHalo_v13',
  'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
  'HLT_GlobalRunHPDNoise_v8',
  'HLT_L1SingleMuOpen_AntiBPTX_v7',
  'HLT_L1TrackerCosmics_v7',
  'HLT_LogMonitor_v4',
  'HLT_Mu15_eta2p1_v5',
  'HLT_PABTagMu_Jet20_Mu4_v1',
  'HLT_PABptxMinusNotBptxPlus_v1',
  'HLT_PABptxPlusNotBptxMinus_v1',
  'HLT_PACastorEmNotHfCoincidencePm_v1',
  'HLT_PACastorEmNotHfSingleChannel_v1',
  'HLT_PACastorEmTotemLowMultiplicity_v1',
  'HLT_PADimuon0_NoVertexing_v1',
  'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleJet20_ForwardBackward_v1',
  'HLT_PADoubleMu4_Acoplanarity03_v1',
  'HLT_PAExclDijet35_HFAND_v1',
  'HLT_PAExclDijet35_HFOR_v1',
  'HLT_PAForJet100Eta2_v1',
  'HLT_PAForJet100Eta3_v1',
  'HLT_PAForJet20Eta2_v1',
  'HLT_PAForJet20Eta3_v1',
  'HLT_PAForJet40Eta2_v1',
  'HLT_PAForJet40Eta3_v1',
  'HLT_PAForJet60Eta2_v1',
  'HLT_PAForJet60Eta3_v1',
  'HLT_PAForJet80Eta2_v1',
  'HLT_PAForJet80Eta3_v1',
  'HLT_PAFullTrack12_v2',
  'HLT_PAFullTrack20_v2',
  'HLT_PAFullTrack30_v2',
  'HLT_PAFullTrack50_v2',
  'HLT_PAHFOR_SingleTrack_v1',
  'HLT_PAHFSumET100_v3',
  'HLT_PAHFSumET140_v3',
  'HLT_PAHFSumET170_v3',
  'HLT_PAHFSumET210_v3',
  'HLT_PAHcalNZS_v1',
  'HLT_PAHcalPhiSym_v1',
  'HLT_PAHcalUTCA_v1',
  'HLT_PAJet100_NoJetID_v1',
  'HLT_PAJet120_NoJetID_v1',
  'HLT_PAJet20_NoJetID_v1',
  'HLT_PAJet40ETM30_v1',
  'HLT_PAJet40_NoJetID_v1',
  'HLT_PAJet60ETM30_v1',
  'HLT_PAJet60_NoJetID_v1',
  'HLT_PAJet80_NoJetID_v1',
  'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
  'HLT_PAL1DoubleEG3_FwdVeto_v1',
  'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
  'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
  'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
  'HLT_PAL1DoubleMu0_HighQ_v1',
  'HLT_PAL1DoubleMu0_v1',
  'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
  'HLT_PAL1DoubleMuOpen_v1',
  'HLT_PAL1SingleEG20_TotemDiffractive_v1',
  'HLT_PAL1SingleJet16_v1',
  'HLT_PAL1SingleJet36_v1',
  'HLT_PAL1SingleJet52_TotemDiffractive_v1',
  'HLT_PAL1SingleMu20_TotemDiffractive_v1',
  'HLT_PAL1Tech53_MB_SingleTrack_v1',
  'HLT_PAL1Tech53_MB_v1',
  'HLT_PAL1Tech54_ZeroBias_v1',
  'HLT_PAL1Tech63_CASTORHaloMuon_v1',
  'HLT_PAL1Tech_HBHEHO_totalOR_v1',
  'HLT_PAL2DoubleMu3_v1',
  'HLT_PAMinBiasBHC_OR_v1',
  'HLT_PAMinBiasBHC_v1',
  'HLT_PAMinBiasHF_OR_v1',
  'HLT_PAMinBiasHF_v1',
  'HLT_PAMinBiasHfOrBHC_v1',
  'HLT_PAMu12_v1',
  'HLT_PAMu3PFJet20_v1',
  'HLT_PAMu3PFJet40_v1',
  'HLT_PAMu3_v1',
  'HLT_PAMu7PFJet20_v1',
  'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
  'HLT_PAMu7_v1',
  'HLT_PAPhoton10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton20_TightCaloIdVL_v1',
  'HLT_PAPhoton30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton30_TightCaloIdVL_v1',
  'HLT_PAPhoton40_NoCaloIdVL_v1',
  'HLT_PAPhoton40_TightCaloIdVL_v1',
  'HLT_PAPhoton60_NoCaloIdVL_v1',
  'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
  'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
  'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
  'HLT_PAPixelTracks_Multiplicity100_v2',
  'HLT_PAPixelTracks_Multiplicity130_v2',
  'HLT_PAPixelTracks_Multiplicity160_v2',
  'HLT_PAPixelTracks_Multiplicity190_v2',
  'HLT_PAPixelTracks_Multiplicity220_v2',
  'HLT_PARandom_v1',
  'HLT_PARomanPots_Tech52_v1',
  'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleForJet15_v1',
  'HLT_PASingleForJet25_v1',
  'HLT_PAT1minbias_Tech55_v1',
  'HLT_PATech35_HFSumET100_v3',
  'HLT_PATech35_v1',
  'HLT_PATripleJet100_20_20_v1',
  'HLT_PATripleJet20_20_20_v1',
  'HLT_PATripleJet40_20_20_v1',
  'HLT_PATripleJet60_20_20_v1',
  'HLT_PATripleJet80_20_20_v1',
  'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
  'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
  'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
  'HLT_PAZeroBiasPixel_DoubleTrack_v1',
  'HLT_PAZeroBiasPixel_SingleTrack_v1',
  'HLT_PAZeroBias_v1',
  'HLT_PPL1DoubleJetC36_v1',
  'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
  'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
  'HLT_PPPixelTracks_Multiplicity55_v1',
  'HLT_PPPixelTracks_Multiplicity70_v1',
  'HLT_PPPixelTracks_Multiplicity85_v1',
  'HLT_Physics_v5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputALCAP0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAP0.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'AlCa_PAEcalEtaEBonly_v1',
  'AlCa_PAEcalEtaEEonly_v1',
  'AlCa_PAEcalPi0EBonly_v1',
  'AlCa_PAEcalPi0EEonly_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlCaEtaEBUncalibrator_*_*',
      'keep *_hltAlCaEtaEEUncalibrator_*_*',
      'keep *_hltAlCaEtaRecHitsFilterEEonly_etaEcalRecHitsES_*',
      'keep *_hltAlCaPi0EBUncalibrator_*_*',
      'keep *_hltAlCaPi0EEUncalibrator_*_*',
      'keep *_hltAlCaPi0RecHitsFilterEEonly_pi0EcalRecHitsES_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCAPHISYM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAPHISYM.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'AlCa_EcalPhiSym_v13' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlCaPhiSymUncalibrator_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputALCALUMIPIXELS = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALUMIPIXELS.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'AlCa_LumiPixels_Random_v1',
  'AlCa_LumiPixels_ZeroBias_v4',
  'AlCa_LumiPixels_v8' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelectorLumiPixels_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_DTCalibration_v2',
  'HLT_EcalCalibration_v3',
  'HLT_HcalCalibration_v3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltDTCalibrationRaw_*_*',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep *_hltHcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQM.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_Activity_Ecal_SC7_v13',
  'HLT_BeamGas_HF_Beam1_v5',
  'HLT_BeamGas_HF_Beam2_v5',
  'HLT_BeamHalo_v13',
  'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
  'HLT_GlobalRunHPDNoise_v8',
  'HLT_L1SingleMuOpen_AntiBPTX_v7',
  'HLT_L1TrackerCosmics_v7',
  'HLT_Mu15_eta2p1_v5',
  'HLT_PABTagMu_Jet20_Mu4_v1',
  'HLT_PABptxMinusNotBptxPlus_v1',
  'HLT_PABptxPlusNotBptxMinus_v1',
  'HLT_PACastorEmNotHfCoincidencePm_v1',
  'HLT_PACastorEmNotHfSingleChannel_v1',
  'HLT_PACastorEmTotemLowMultiplicity_v1',
  'HLT_PADimuon0_NoVertexing_v1',
  'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleJet20_ForwardBackward_v1',
  'HLT_PADoubleMu4_Acoplanarity03_v1',
  'HLT_PAExclDijet35_HFAND_v1',
  'HLT_PAExclDijet35_HFOR_v1',
  'HLT_PAForJet100Eta2_v1',
  'HLT_PAForJet100Eta3_v1',
  'HLT_PAForJet20Eta2_v1',
  'HLT_PAForJet20Eta3_v1',
  'HLT_PAForJet40Eta2_v1',
  'HLT_PAForJet40Eta3_v1',
  'HLT_PAForJet60Eta2_v1',
  'HLT_PAForJet60Eta3_v1',
  'HLT_PAForJet80Eta2_v1',
  'HLT_PAForJet80Eta3_v1',
  'HLT_PAFullTrack12_v2',
  'HLT_PAFullTrack20_v2',
  'HLT_PAFullTrack30_v2',
  'HLT_PAFullTrack50_v2',
  'HLT_PAHFOR_SingleTrack_v1',
  'HLT_PAHFSumET100_v3',
  'HLT_PAHFSumET140_v3',
  'HLT_PAHFSumET170_v3',
  'HLT_PAHFSumET210_v3',
  'HLT_PAHcalNZS_v1',
  'HLT_PAHcalPhiSym_v1',
  'HLT_PAHcalUTCA_v1',
  'HLT_PAJet100_NoJetID_v1',
  'HLT_PAJet120_NoJetID_v1',
  'HLT_PAJet20_NoJetID_v1',
  'HLT_PAJet40ETM30_v1',
  'HLT_PAJet40_NoJetID_v1',
  'HLT_PAJet60ETM30_v1',
  'HLT_PAJet60_NoJetID_v1',
  'HLT_PAJet80_NoJetID_v1',
  'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
  'HLT_PAL1DoubleEG3_FwdVeto_v1',
  'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
  'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
  'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
  'HLT_PAL1DoubleMu0_HighQ_v1',
  'HLT_PAL1DoubleMu0_v1',
  'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
  'HLT_PAL1DoubleMuOpen_v1',
  'HLT_PAL1SingleEG20_TotemDiffractive_v1',
  'HLT_PAL1SingleJet16_v1',
  'HLT_PAL1SingleJet36_v1',
  'HLT_PAL1SingleJet52_TotemDiffractive_v1',
  'HLT_PAL1SingleMu20_TotemDiffractive_v1',
  'HLT_PAL1Tech53_MB_SingleTrack_v1',
  'HLT_PAL1Tech53_MB_v1',
  'HLT_PAL1Tech54_ZeroBias_v1',
  'HLT_PAL1Tech63_CASTORHaloMuon_v1',
  'HLT_PAL1Tech_HBHEHO_totalOR_v1',
  'HLT_PAL2DoubleMu3_v1',
  'HLT_PAMinBiasBHC_OR_v1',
  'HLT_PAMinBiasBHC_v1',
  'HLT_PAMinBiasHF_OR_v1',
  'HLT_PAMinBiasHF_v1',
  'HLT_PAMinBiasHfOrBHC_v1',
  'HLT_PAMu12_v1',
  'HLT_PAMu3PFJet20_v1',
  'HLT_PAMu3PFJet40_v1',
  'HLT_PAMu3_v1',
  'HLT_PAMu7PFJet20_v1',
  'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
  'HLT_PAMu7_v1',
  'HLT_PAPhoton10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton20_TightCaloIdVL_v1',
  'HLT_PAPhoton30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton30_TightCaloIdVL_v1',
  'HLT_PAPhoton40_NoCaloIdVL_v1',
  'HLT_PAPhoton40_TightCaloIdVL_v1',
  'HLT_PAPhoton60_NoCaloIdVL_v1',
  'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
  'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
  'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
  'HLT_PAPixelTracks_Multiplicity100_v2',
  'HLT_PAPixelTracks_Multiplicity130_v2',
  'HLT_PAPixelTracks_Multiplicity160_v2',
  'HLT_PAPixelTracks_Multiplicity190_v2',
  'HLT_PAPixelTracks_Multiplicity220_v2',
  'HLT_PARandom_v1',
  'HLT_PARomanPots_Tech52_v1',
  'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleForJet15_v1',
  'HLT_PASingleForJet25_v1',
  'HLT_PAT1minbias_Tech55_v1',
  'HLT_PATech35_HFSumET100_v3',
  'HLT_PATech35_v1',
  'HLT_PATripleJet100_20_20_v1',
  'HLT_PATripleJet20_20_20_v1',
  'HLT_PATripleJet40_20_20_v1',
  'HLT_PATripleJet60_20_20_v1',
  'HLT_PATripleJet80_20_20_v1',
  'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
  'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
  'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
  'HLT_PAZeroBiasPixel_DoubleTrack_v1',
  'HLT_PAZeroBiasPixel_SingleTrack_v1',
  'HLT_PAZeroBias_v1',
  'HLT_PPL1DoubleJetC36_v1',
  'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
  'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
  'HLT_PPPixelTracks_Multiplicity55_v1',
  'HLT_PPPixelTracks_Multiplicity70_v1',
  'HLT_PPPixelTracks_Multiplicity85_v1',
  'HLT_Physics_v5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputEcalCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputEcalCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_EcalCalibration_v3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputExpress = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputExpress.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
  'HLT_Mu15_eta2p1_v5',
  'HLT_PABTagMu_Jet20_Mu4_v1',
  'HLT_PABptxMinusNotBptxPlus_v1',
  'HLT_PABptxPlusNotBptxMinus_v1',
  'HLT_PACastorEmNotHfCoincidencePm_v1',
  'HLT_PACastorEmNotHfSingleChannel_v1',
  'HLT_PACastorEmTotemLowMultiplicity_v1',
  'HLT_PADimuon0_NoVertexing_v1',
  'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleJet20_ForwardBackward_v1',
  'HLT_PADoubleMu4_Acoplanarity03_v1',
  'HLT_PAExclDijet35_HFAND_v1',
  'HLT_PAExclDijet35_HFOR_v1',
  'HLT_PAForJet100Eta2_v1',
  'HLT_PAForJet100Eta3_v1',
  'HLT_PAForJet20Eta2_v1',
  'HLT_PAForJet20Eta3_v1',
  'HLT_PAForJet40Eta2_v1',
  'HLT_PAForJet40Eta3_v1',
  'HLT_PAForJet60Eta2_v1',
  'HLT_PAForJet60Eta3_v1',
  'HLT_PAForJet80Eta2_v1',
  'HLT_PAForJet80Eta3_v1',
  'HLT_PAFullTrack12_v2',
  'HLT_PAFullTrack20_v2',
  'HLT_PAFullTrack30_v2',
  'HLT_PAFullTrack50_v2',
  'HLT_PAHFOR_SingleTrack_v1',
  'HLT_PAHFSumET100_v3',
  'HLT_PAHFSumET140_v3',
  'HLT_PAHFSumET170_v3',
  'HLT_PAHFSumET210_v3',
  'HLT_PAJet100_NoJetID_v1',
  'HLT_PAJet120_NoJetID_v1',
  'HLT_PAJet20_NoJetID_v1',
  'HLT_PAJet40ETM30_v1',
  'HLT_PAJet40_NoJetID_v1',
  'HLT_PAJet60ETM30_v1',
  'HLT_PAJet60_NoJetID_v1',
  'HLT_PAJet80_NoJetID_v1',
  'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
  'HLT_PAL1DoubleEG3_FwdVeto_v1',
  'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
  'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
  'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
  'HLT_PAL1DoubleMu0_HighQ_v1',
  'HLT_PAL1DoubleMu0_v1',
  'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
  'HLT_PAL1DoubleMuOpen_v1',
  'HLT_PAL1SingleEG20_TotemDiffractive_v1',
  'HLT_PAL1SingleJet16_v1',
  'HLT_PAL1SingleJet36_v1',
  'HLT_PAL1SingleJet52_TotemDiffractive_v1',
  'HLT_PAL1SingleMu20_TotemDiffractive_v1',
  'HLT_PAL1Tech53_MB_SingleTrack_v1',
  'HLT_PAL1Tech53_MB_v1',
  'HLT_PAL1Tech54_ZeroBias_v1',
  'HLT_PAL1Tech63_CASTORHaloMuon_v1',
  'HLT_PAL1Tech_HBHEHO_totalOR_v1',
  'HLT_PAL2DoubleMu3_v1',
  'HLT_PAMinBiasBHC_OR_v1',
  'HLT_PAMinBiasBHC_v1',
  'HLT_PAMinBiasHF_OR_v1',
  'HLT_PAMinBiasHF_v1',
  'HLT_PAMinBiasHfOrBHC_v1',
  'HLT_PAMu12_v1',
  'HLT_PAMu3PFJet20_v1',
  'HLT_PAMu3PFJet40_v1',
  'HLT_PAMu3_v1',
  'HLT_PAMu7PFJet20_v1',
  'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
  'HLT_PAMu7_v1',
  'HLT_PAPhoton10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton20_TightCaloIdVL_v1',
  'HLT_PAPhoton30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton30_TightCaloIdVL_v1',
  'HLT_PAPhoton40_NoCaloIdVL_v1',
  'HLT_PAPhoton40_TightCaloIdVL_v1',
  'HLT_PAPhoton60_NoCaloIdVL_v1',
  'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
  'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
  'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
  'HLT_PAPixelTracks_Multiplicity100_v2',
  'HLT_PAPixelTracks_Multiplicity130_v2',
  'HLT_PAPixelTracks_Multiplicity160_v2',
  'HLT_PAPixelTracks_Multiplicity190_v2',
  'HLT_PAPixelTracks_Multiplicity220_v2',
  'HLT_PARandom_v1',
  'HLT_PARomanPots_Tech52_v1',
  'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleForJet15_v1',
  'HLT_PASingleForJet25_v1',
  'HLT_PAT1minbias_Tech55_v1',
  'HLT_PATech35_HFSumET100_v3',
  'HLT_PATech35_v1',
  'HLT_PATripleJet100_20_20_v1',
  'HLT_PATripleJet20_20_20_v1',
  'HLT_PATripleJet40_20_20_v1',
  'HLT_PATripleJet60_20_20_v1',
  'HLT_PATripleJet80_20_20_v1',
  'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
  'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
  'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
  'HLT_PAZeroBiasPixel_DoubleTrack_v1',
  'HLT_PAZeroBiasPixel_SingleTrack_v1',
  'HLT_PAZeroBias_v1',
  'HLT_PPL1DoubleJetC36_v1',
  'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
  'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
  'HLT_PPPixelTracks_Multiplicity55_v1',
  'HLT_PPPixelTracks_Multiplicity70_v1',
  'HLT_PPPixelTracks_Multiplicity85_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputHLTDQM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHLTDQM.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_Ele22_CaloIdL_CaloIsoVL_v6',
  'HLT_Mu15_eta2p1_v5',
  'HLT_PABTagMu_Jet20_Mu4_v1',
  'HLT_PABptxMinusNotBptxPlus_v1',
  'HLT_PABptxPlusNotBptxMinus_v1',
  'HLT_PACastorEmNotHfCoincidencePm_v1',
  'HLT_PACastorEmNotHfSingleChannel_v1',
  'HLT_PACastorEmTotemLowMultiplicity_v1',
  'HLT_PADimuon0_NoVertexing_v1',
  'HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1',
  'HLT_PADoubleJet20_ForwardBackward_v1',
  'HLT_PADoubleMu4_Acoplanarity03_v1',
  'HLT_PAExclDijet35_HFAND_v1',
  'HLT_PAExclDijet35_HFOR_v1',
  'HLT_PAForJet100Eta2_v1',
  'HLT_PAForJet100Eta3_v1',
  'HLT_PAForJet20Eta2_v1',
  'HLT_PAForJet20Eta3_v1',
  'HLT_PAForJet40Eta2_v1',
  'HLT_PAForJet40Eta3_v1',
  'HLT_PAForJet60Eta2_v1',
  'HLT_PAForJet60Eta3_v1',
  'HLT_PAForJet80Eta2_v1',
  'HLT_PAForJet80Eta3_v1',
  'HLT_PAFullTrack12_v2',
  'HLT_PAFullTrack20_v2',
  'HLT_PAFullTrack30_v2',
  'HLT_PAFullTrack50_v2',
  'HLT_PAHFOR_SingleTrack_v1',
  'HLT_PAHFSumET100_v3',
  'HLT_PAHFSumET140_v3',
  'HLT_PAHFSumET170_v3',
  'HLT_PAHFSumET210_v3',
  'HLT_PAHcalNZS_v1',
  'HLT_PAHcalPhiSym_v1',
  'HLT_PAHcalUTCA_v1',
  'HLT_PAJet100_NoJetID_v1',
  'HLT_PAJet120_NoJetID_v1',
  'HLT_PAJet20_NoJetID_v1',
  'HLT_PAJet40ETM30_v1',
  'HLT_PAJet40_NoJetID_v1',
  'HLT_PAJet60ETM30_v1',
  'HLT_PAJet60_NoJetID_v1',
  'HLT_PAJet80_NoJetID_v1',
  'HLT_PAL1CastorTotalTotemLowMultiplicity_v1',
  'HLT_PAL1DoubleEG3_FwdVeto_v1',
  'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PAL1DoubleEG5_TotemDiffractive_v1',
  'HLT_PAL1DoubleJet20_TotemDiffractive_v1',
  'HLT_PAL1DoubleJetC36_TotemDiffractive_v1',
  'HLT_PAL1DoubleMu0_HighQ_v1',
  'HLT_PAL1DoubleMu0_v1',
  'HLT_PAL1DoubleMu5_TotemDiffractive_v1',
  'HLT_PAL1DoubleMuOpen_v1',
  'HLT_PAL1SingleEG20_TotemDiffractive_v1',
  'HLT_PAL1SingleJet16_v1',
  'HLT_PAL1SingleJet36_v1',
  'HLT_PAL1SingleJet52_TotemDiffractive_v1',
  'HLT_PAL1SingleMu20_TotemDiffractive_v1',
  'HLT_PAL1Tech53_MB_SingleTrack_v1',
  'HLT_PAL1Tech53_MB_v1',
  'HLT_PAL1Tech54_ZeroBias_v1',
  'HLT_PAL1Tech63_CASTORHaloMuon_v1',
  'HLT_PAL1Tech_HBHEHO_totalOR_v1',
  'HLT_PAL2DoubleMu3_v1',
  'HLT_PAMinBiasBHC_OR_v1',
  'HLT_PAMinBiasBHC_v1',
  'HLT_PAMinBiasHF_OR_v1',
  'HLT_PAMinBiasHF_v1',
  'HLT_PAMinBiasHfOrBHC_v1',
  'HLT_PAMu12_v1',
  'HLT_PAMu3PFJet20_v1',
  'HLT_PAMu3PFJet40_v1',
  'HLT_PAMu3_v1',
  'HLT_PAMu7PFJet20_v1',
  'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1',
  'HLT_PAMu7_v1',
  'HLT_PAPhoton10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton10_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1',
  'HLT_PAPhoton15_Photon10_TightCaloIdVL_v1',
  'HLT_PAPhoton15_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1',
  'HLT_PAPhoton20_Photon15_TightCaloIdVL_v1',
  'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1',
  'HLT_PAPhoton20_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton20_TightCaloIdVL_v1',
  'HLT_PAPhoton30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1',
  'HLT_PAPhoton30_TightCaloIdVL_Iso50_v1',
  'HLT_PAPhoton30_TightCaloIdVL_v1',
  'HLT_PAPhoton40_NoCaloIdVL_v1',
  'HLT_PAPhoton40_TightCaloIdVL_v1',
  'HLT_PAPhoton60_NoCaloIdVL_v1',
  'HLT_PAPixelTrackMultiplicity100_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1',
  'HLT_PAPixelTrackMultiplicity130_FullTrack12_v2',
  'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2',
  'HLT_PAPixelTrackMultiplicity160_FullTrack12_v2',
  'HLT_PAPixelTracks_Multiplicity100_v2',
  'HLT_PAPixelTracks_Multiplicity130_v2',
  'HLT_PAPixelTracks_Multiplicity160_v2',
  'HLT_PAPixelTracks_Multiplicity190_v2',
  'HLT_PAPixelTracks_Multiplicity220_v2',
  'HLT_PARandom_v1',
  'HLT_PARomanPots_Tech52_v1',
  'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleEle6_CaloIdT_TrkIdVL_v1',
  'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1',
  'HLT_PASingleForJet15_v1',
  'HLT_PASingleForJet25_v1',
  'HLT_PAT1minbias_Tech55_v1',
  'HLT_PATech35_HFSumET100_v3',
  'HLT_PATech35_v1',
  'HLT_PATripleJet100_20_20_v1',
  'HLT_PATripleJet20_20_20_v1',
  'HLT_PATripleJet40_20_20_v1',
  'HLT_PATripleJet60_20_20_v1',
  'HLT_PATripleJet80_20_20_v1',
  'HLT_PAUpcSingleEG5Full_TrackVeto7_v1',
  'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1',
  'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1',
  'HLT_PAUpcSingleMuOpenTkMu_Onia_v1',
  'HLT_PAZeroBiasPixel_DoubleTrack_v1',
  'HLT_PAZeroBiasPixel_SingleTrack_v1',
  'HLT_PAZeroBias_v1',
  'HLT_PPL1DoubleJetC36_v1',
  'HLT_PPPixelTrackMultiplicity55_FullTrack12_v1',
  'HLT_PPPixelTrackMultiplicity70_FullTrack12_v1',
  'HLT_PPPixelTracks_Multiplicity55_v1',
  'HLT_PPPixelTracks_Multiplicity70_v1',
  'HLT_PPPixelTracks_Multiplicity85_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltTriggerSummaryAOD_*_*',
      'keep DcsStatuss_hltScalersRawToDigi_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep LumiScalerss_hltScalersRawToDigi_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputNanoDST = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputNanoDST.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'DST_Physics_v5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelector_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep L1MuGMTReadoutCollection_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputRPCMON = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputRPCMON.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'AlCa_RPCMuonNoHits_v9',
  'AlCa_RPCMuonNoTriggers_v9',
  'AlCa_RPCMuonNormalisation_v9' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltCscSegments_*_*',
      'keep *_hltDt4DSegments_*_*',
      'keep *_hltMuonCSCDigis_MuonCSCStripDigi_*',
      'keep *_hltMuonCSCDigis_MuonCSCWireDigi_*',
      'keep *_hltMuonDTDigis_*_*',
      'keep *_hltMuonRPCDigis_*_*',
      'keep *_hltRpcRecHits_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep L1MuGMTCands_hltGtDigis_*_*',
      'keep L1MuGMTReadoutCollection_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputTrackerCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputTrackerCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_TrackerCalibration_v3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltTrackerCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)

process.HLTL1UnpackerSequence = cms.Sequence( process.hltGtDigis + process.hltGctDigis + process.hltL1GtObjectMap + process.hltL1extraParticles )
process.HLTBeamSpot = cms.Sequence( process.hltScalersRawToDigi + process.hltOnlineBeamSpot )
process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTEcalActivitySequence = cms.Sequence( process.hltEcalRawToRecHitFacility + process.hltESRawToRecHitFacility + process.hltEcalRegionalRestFEDs + process.hltEcalRegionalESRestFEDs + process.hltEcalRecHitAll + process.hltESRecHitAll + process.hltHybridSuperClustersActivity + process.hltCorrectedHybridSuperClustersActivity + process.hltMulti5x5BasicClustersActivity + process.hltMulti5x5SuperClustersActivity + process.hltMulti5x5SuperClustersWithPreshowerActivity + process.hltCorrectedMulti5x5SuperClustersWithPreshowerActivity + process.hltRecoEcalSuperClusterActivityCandidate + process.hltEcalActivitySuperClusterWrapper )
process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )
process.HLTMuonLocalRecoSequence = cms.Sequence( process.hltMuonDTDigis + process.hltDt1DRecHits + process.hltDt4DSegments + process.hltMuonCSCDigis + process.hltCsc2DRecHits + process.hltCscSegments + process.hltMuonRPCDigis + process.hltRpcRecHits )
process.HLTL2muonrecoNocandSequence = cms.Sequence( process.HLTMuonLocalRecoSequence + process.hltL2OfflineMuonSeeds + process.hltL2MuonSeeds + process.hltL2Muons )
process.HLTL2muonrecoSequence = cms.Sequence( process.HLTL2muonrecoNocandSequence + process.hltL2MuonCandidates )
process.HLTDoLocalPixelSequence = cms.Sequence( process.hltSiPixelDigis + process.hltSiPixelClusters + process.hltSiPixelRecHits )
process.HLTDoLocalStripSequence = cms.Sequence( process.hltSiStripExcludedFEDListProducer + process.hltSiStripRawToClustersFacility + process.hltSiStripClusters )
process.HLTL3muonTkCandidateSequence = cms.Sequence( process.HLTDoLocalPixelSequence + process.HLTDoLocalStripSequence + process.hltL3TrajSeedOIState + process.hltL3TrackCandidateFromL2OIState + process.hltL3TkTracksFromL2OIState + process.hltL3MuonsOIState + process.hltL3TrajSeedOIHit + process.hltL3TrackCandidateFromL2OIHit + process.hltL3TkTracksFromL2OIHit + process.hltL3MuonsOIHit + process.hltL3TkFromL2OICombination + process.hltL3TrajSeedIOHit + process.hltL3TrackCandidateFromL2IOHit + process.hltL3TkTracksFromL2IOHit + process.hltL3MuonsIOHit + process.hltL3TrajectorySeed + process.hltL3TrackCandidateFromL2 )
process.HLTL3muonrecoNocandSequence = cms.Sequence( process.HLTL3muonTkCandidateSequence + process.hltL3TkTracksFromL2 + process.hltL3MuonsLinksCombination + process.hltL3Muons )
process.HLTL3muonrecoSequence = cms.Sequence( process.HLTL3muonrecoNocandSequence + process.hltL3MuonCandidates )
process.HLTDoRegionalEgammaEcalSequence = cms.Sequence( process.hltESRawToRecHitFacility + process.hltEcalRawToRecHitFacility + process.hltEcalRegionalEgammaFEDs + process.hltEcalRegionalEgammaRecHit + process.hltESRegionalEgammaRecHit )
process.HLTMulti5x5SuperClusterL1Seeded = cms.Sequence( process.hltMulti5x5BasicClustersL1Seeded + process.hltMulti5x5SuperClustersL1Seeded + process.hltMulti5x5EndcapSuperClustersWithPreshowerL1Seeded + process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Seeded )
process.HLTL1SeededEcalClustersSequence = cms.Sequence( process.hltHybridSuperClustersL1Seeded + process.hltCorrectedHybridSuperClustersL1Seeded + process.HLTMulti5x5SuperClusterL1Seeded )
process.HLTDoLocalHcalWithoutHOSequence = cms.Sequence( process.hltHcalDigis + process.hltHbhereco + process.hltHfreco )
process.HLTEle22CaloIdLCaloIsoVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG12 + process.hltEG22EtFilter + process.hltL1SeededHLTClusterShape + process.hltEG22CaloIdLClusterShapeFilter + process.hltL1SeededPhotonEcalIso + process.hltEG22CaloIdLCaloIsoVLEcalIsoFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltEG22CaloIdLCaloIsoVLHEFilter + process.hltL1SeededPhotonHcalIso + process.hltEG22CaloIdLCaloIsoVLHcalIsoFilter + process.HLTDoLocalPixelSequence + process.HLTDoLocalStripSequence + process.hltL1SeededStartUpElectronPixelSeeds + process.hltEle22CaloIdLCaloIsoVLPixelMatchFilter )
process.HLTDoLocalPixelClustersSequence = cms.Sequence( process.hltSiPixelDigis + process.hltSiPixelClusters )
process.HLTPABeginSequenceUTCA = cms.Sequence( process.hltTriggerType + process.hltPAL1EventNumberUTCA + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTBeginSequenceNZS = cms.Sequence( process.hltTriggerType + process.hltL1EventNumberNZS + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTBeginSequenceCalibration = cms.Sequence( process.hltCalibrationEventsFilter + process.hltGtDigis )
process.HLTBeginSequenceAntiBPTX = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.hltBPTXAntiCoincidence + process.HLTBeamSpot )
process.HLTDoRegionalPi0EtaSequence = cms.Sequence( process.hltESRawToRecHitFacility + process.hltEcalRawToRecHitFacility + process.hltEcalRegionalPi0EtaFEDs + process.hltESRegionalPi0EtaRecHit + process.hltEcalRegionalPi0EtaRecHit )
process.HLTBeginSequenceRandom = cms.Sequence( process.hltRandomEventsFilter + process.hltGtDigis )
process.HLTBeginSequenceBPTX = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.hltBPTXCoincidence + process.HLTBeamSpot )
process.HLTDoLocalHcalSequence = cms.Sequence( process.hltHcalDigis + process.hltHbhereco + process.hltHfreco + process.hltHoreco )
process.HLTDoCaloSequence = cms.Sequence( process.hltEcalRawToRecHitFacility + process.hltEcalRegionalRestFEDs + process.hltEcalRecHitAll + process.HLTDoLocalHcalSequence + process.hltTowerMakerForAll )
process.HLTRecoJetSequenceAK5Uncorrected = cms.Sequence( process.HLTDoCaloSequence + process.hltAntiKT5CaloJets )
process.HLTRecoJetSequenceAK5Corrected = cms.Sequence( process.HLTRecoJetSequenceAK5Uncorrected + process.hltCaloJetIDPassed + process.hltCaloJetCorrected )
process.HLTDoRegionalJetEcalSequence = cms.Sequence( process.hltEcalRawToRecHitFacility + process.hltEcalRegionalJetsFEDs + process.hltEcalRegionalJetsRecHit )
process.HLTRegionalTowerMakerForJetsSequence = cms.Sequence( process.HLTDoRegionalJetEcalSequence + process.HLTDoLocalHcalSequence + process.hltTowerMakerForJets )
process.HLTDoLocalHfSequence = cms.Sequence( process.hltHcalDigis + process.hltHfreco + process.hltTowerMakerForHf )
process.HLTRecoJetSequenceAK5L1FastJetCorrected = cms.Sequence( process.HLTDoCaloSequence + process.hltKT6CaloJets + process.hltAntiKT5CaloJets + process.hltCaloJetIDPassed + process.hltCaloJetL1FastJetCorrected )
process.HLTBTagMuJet20L1FastJetSequenceL25BPH = cms.Sequence( process.HLTL2muonrecoNocandSequence + process.hltBSoftMuonGetJetsFromJet20L1FastJetBPH + process.hltSelector4JetsJet20L1FastJetBPH + process.hltBSoftMuonJet20L1FastJetL25JetsBPH + process.hltBSoftMuonJet20L1FastJetL25TagInfosBPH + process.hltBSoftMuonJet20L1FastJetL25BJetTagsByDRBPH )
process.HLTBTagMuJet20L1FastJetMu5SelSequenceL3BPH = cms.Sequence( process.HLTL3muonrecoNocandSequence + process.hltBSoftMuonMu4L3 + process.hltBSoftMuonJet20L1FastJetMu5SelL3TagInfosBPH + process.hltBSoftMuonJet20L1FastJetMu5SelL3BJetTagsByDRBPH )
process.HLTDoCaloSequencePF = cms.Sequence( process.hltEcalRawToRecHitFacility + process.hltEcalRegionalRestFEDs + process.hltEcalRecHitAll + process.HLTDoLocalHcalSequence + process.hltTowerMakerForPF )
process.HLTRecoJetSequenceAK5UncorrectedPF = cms.Sequence( process.HLTDoCaloSequencePF + process.hltAntiKT5CaloJetsPF )
process.HLTRecoJetSequencePrePF = cms.Sequence( process.HLTRecoJetSequenceAK5UncorrectedPF + process.hltAntiKT5CaloJetsPFEt5 )
process.HLTRecopixelvertexingSequence = cms.Sequence( process.hltPixelTracks + process.hltPixelVertices )
process.HLTIterativeTrackingIteration0 = cms.Sequence( process.hltPFJetPixelSeedsFromPixelTracks + process.hltPFJetCkfTrackCandidates + process.hltPFJetCtfWithMaterialTracks + process.hltPFlowTrackSelectionHighPurity + process.hltTrackRefsForJetsIter0 + process.hltAntiKT5TrackJetsIter0 + process.hltTrackAndTauJetsIter0 )
process.HLTIterativeTrackingIteration1 = cms.Sequence( process.hltIter1ClustersRefRemoval + process.hltIter1SiStripClusters + process.hltIter1PFJetPixelSeeds + process.hltIter1PFJetCkfTrackCandidates + process.hltIter1PFJetCtfWithMaterialTracks + process.hltIter1PFlowTrackSelectionHighPurityLoose + process.hltIter1PFlowTrackSelectionHighPurityTight + process.hltIter1PFlowTrackSelectionHighPurity + process.hltIter1Merged + process.hltTrackRefsForJetsIter1 + process.hltAntiKT5TrackJetsIter1 + process.hltTrackAndTauJetsIter1 )
process.HLTIterativeTrackingIteration2 = cms.Sequence( process.hltIter2ClustersRefRemoval + process.hltIter2SiStripClusters + process.hltIter2PFJetPixelSeeds + process.hltIter2PFJetCkfTrackCandidates + process.hltIter2PFJetCtfWithMaterialTracks + process.hltIter2PFlowTrackSelectionHighPurity + process.hltIter2Merged + process.hltTrackRefsForJetsIter2 + process.hltAntiKT5TrackJetsIter2 + process.hltTrackAndTauJetsIter2 )
process.HLTIterativeTrackingIteration3 = cms.Sequence( process.hltIter3ClustersRefRemoval + process.hltIter3SiStripClusters + process.hltIter3PFJetMixedSeeds + process.hltIter3PFJetCkfTrackCandidates + process.hltIter3PFJetCtfWithMaterialTracks + process.hltIter3PFlowTrackSelectionHighPurityLoose + process.hltIter3PFlowTrackSelectionHighPurityTight + process.hltIter3PFlowTrackSelectionHighPurity + process.hltIter3Merged + process.hltTrackRefsForJetsIter3 + process.hltAntiKT5TrackJetsIter3 + process.hltTrackAndTauJetsIter3 )
process.HLTIterativeTrackingIteration4 = cms.Sequence( process.hltIter4ClustersRefRemoval + process.hltIter4SiStripClusters + process.hltIter4PFJetPixelLessSeeds + process.hltIter4PFJetCkfTrackCandidates + process.hltIter4PFJetCtfWithMaterialTracks + process.hltIter4PFlowTrackSelectionHighPurity + process.hltIter4Merged )
process.HLTIterativeTracking = cms.Sequence( process.HLTIterativeTrackingIteration0 + process.HLTIterativeTrackingIteration1 + process.HLTIterativeTrackingIteration2 + process.HLTIterativeTrackingIteration3 + process.HLTIterativeTrackingIteration4 )
process.HLTTrackReconstructionForPF = cms.Sequence( process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingSequence + process.HLTDoLocalStripSequence + process.HLTIterativeTracking + process.hltPFMuonMerging + process.hltMuonLinks + process.hltMuons )
process.HLTPreshowerSequence = cms.Sequence( process.hltESRawToRecHitFacility + process.hltEcalRegionalESRestFEDs + process.hltESRecHitAll )
process.HLTParticleFlowSequence = cms.Sequence( process.HLTPreshowerSequence + process.hltParticleFlowRecHitECAL + process.hltParticleFlowRecHitHCAL + process.hltParticleFlowRecHitPS + process.hltParticleFlowClusterECAL + process.hltParticleFlowClusterHCAL + process.hltParticleFlowClusterHFEM + process.hltParticleFlowClusterHFHAD + process.hltParticleFlowClusterPS + process.hltLightPFTracks + process.hltParticleFlowBlock + process.hltParticleFlow )
process.HLTPFL1FastL2L3JetsSequence = cms.Sequence( process.hltKT6PFJets + process.hltAntiKT5PFJets + process.hltAK5PFJetL1FastL2L3Corrected )
process.HLTPFL1FastL2L3JetTriggerSequence = cms.Sequence( process.HLTL2muonrecoSequence + process.HLTL3muonrecoSequence + process.HLTTrackReconstructionForPF + process.HLTParticleFlowSequence + process.HLTPFL1FastL2L3JetsSequence )
process.HLTPFL1FastL2L3ReconstructionSequence = cms.Sequence( process.HLTRecoJetSequencePrePF + process.HLTPFL1FastL2L3JetTriggerSequence )
process.HLTPhoton10NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG10EtFilter )
process.HLTPhoton15NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG15EtFilter )
process.HLTPhoton20NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG20EtPAFilter )
process.HLTPhoton30NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG12 + process.hltPAEG30EtFilter )
process.HLTPhoton40NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG20 + process.hltPAEG40EtFilter )
process.HLTPhoton60NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG24 + process.hltEG60EtFilter )
process.HLTPAPhoton10TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG10EtFilter + process.hltL1SeededHLTClusterShape + process.hltEG10TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltEG10TightCaloIdVLHEFilter )
process.HLTPAPhoton15TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG15EtFilter + process.hltL1SeededHLTClusterShape + process.hltEG15TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltEG15TightCaloIdVLHEFilter )
process.HLTPAPhoton20TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG20EtPAFilter + process.hltL1SeededHLTClusterShape + process.hltEG20TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltEG20TightCaloIdVLHEFilter )
process.HLTPAPhoton30TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG12 + process.hltPAEG30EtFilter + process.hltL1SeededHLTClusterShape + process.hltPAEG30TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltPAEG30TightCaloIdVLHEFilter )
process.HLTPAPhoton40TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG20 + process.hltPAEG40EtFilter + process.hltL1SeededHLTClusterShape + process.hltPAEG40TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltPAEG40TightCaloIdVLHEFilter )
process.HLTPAPhoton10TightCaloIdVLIso50Sequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG10EtFilter + process.hltL1SeededHLTClusterShape + process.hltEG10TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltEG10TightCaloIdVLHEFilter + process.hltL1SeededPhotonEcalIso + process.hltPAPhoton10CaloIdVLIso50EcalIsoFilter + process.hltL1SeededPhotonHcalIso + process.hltPAPhoton10CaloIdVLIso50HcalIsoFilter )
process.HLTPAPhoton15TightCaloIdVLIso50Sequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG15EtFilter + process.hltL1SeededHLTClusterShape + process.hltEG15TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltEG15TightCaloIdVLHEFilter + process.hltL1SeededPhotonEcalIso + process.hltPAPhoton15CaloIdVLIso50EcalIsoFilter + process.hltL1SeededPhotonHcalIso + process.hltPAPhoton15CaloIdVLIso50HcalIsoFilter )
process.HLTPAPhoton20TightCaloIdVLIso50Sequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG5PA + process.hltEG20EtPAFilter + process.hltL1SeededHLTClusterShape + process.hltEG20TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltEG20TightCaloIdVLHEFilter + process.hltL1SeededPhotonEcalIso + process.hltPAPhoton20CaloIdVLIso50EcalIsoFilter + process.hltL1SeededPhotonHcalIso + process.hltPAPhoton20CaloIdVLIso50HcalIsoFilter )
process.HLTPAPhoton30TightCaloIdVLIso50Sequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1SingleEG12 + process.hltPAEG30EtFilter + process.hltL1SeededHLTClusterShape + process.hltPAEG30TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltPAEG30TightCaloIdVLHEFilter + process.hltL1SeededPhotonEcalIso + process.hltPAPhoton30CaloIdVLIso50EcalIsoFilter + process.hltL1SeededPhotonHcalIso + process.hltPAPhoton30CaloIdVLIso50HcalIsoFilter )
process.HLTDoublePhoton10And10NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble10And10EtFilter )
process.HLTDoublePhoton15And10NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble15And10EtFilterFirst + process.hltEGDouble15And10EtFilterSecond )
process.HLTDoublePhoton20And15NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble20And15EtFilterFirst + process.hltEGDouble20And15EtFilterSecond )
process.HLTDoublePhoton20And20NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble20And20EtFilter )
process.HLTDoublePhoton30And30NoCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble30And30EtFilter )
process.HLTPADoublePhoton10And10TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble10And10EtFilter + process.hltL1SeededHLTClusterShape + process.hltPAEGDouble10And10TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltPAEGDouble10And10CaloIdVLHEFilter )
process.HLTPADoublePhoton10And10TightCaloIdVLIso50Sequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble10And10EtFilter + process.hltL1SeededHLTClusterShape + process.hltPAEGDouble10And10TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltPAEGDouble10And10CaloIdVLHEFilter + process.hltL1SeededPhotonEcalIso + process.hltPAPhoton10AndPhoton10CaloIdVLIso50EcalIsoFilter + process.hltL1SeededPhotonHcalIso + process.hltPAPhoton10AndPhoton10CaloIdVLIso50HcalIsoFilter )
process.HLTPADoublePhoton15And10TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble15And10EtFilterFirst + process.hltEGDouble15And10EtFilterSecond + process.hltL1SeededHLTClusterShape + process.hltPAEGDouble15And10TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltPAEGDouble15And10CaloIdVLHEFilter )
process.HLTPADoublePhoton20And15TightCaloIdVLSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate + process.hltEGRegionalL1DoubleEG5 + process.hltEGDouble20And15EtFilterFirst + process.hltEGDouble20And15EtFilterSecond + process.hltL1SeededHLTClusterShape + process.hltPAEGDouble20And15TightCaloIdVLClusterShapeFilter + process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE + process.hltPAEGDouble20And15CaloIdVLHEFilter )
process.HLTDoEGammaStartupSequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1SeededEcalClustersSequence + process.hltL1SeededRecoEcalCandidate )
process.HLTDoEgammaClusterShapeSequence = cms.Sequence( process.hltL1SeededHLTClusterShape )
process.HLTDoEGammaHESequence = cms.Sequence( process.HLTDoLocalHcalWithoutHOSequence + process.hltL1SeededPhotonHcalForHE )
process.HLTDoEGammaPixelSequence = cms.Sequence( process.HLTDoLocalPixelSequence + process.HLTDoLocalStripSequence + process.hltL1SeededStartUpElectronPixelSeeds )
process.HLTSingleEle6CaloIdTSequence = cms.Sequence( process.HLTDoEGammaStartupSequence + process.hltEGRegionalL1SingleEG5PA + process.hltSingleEG6EtFilterL1SingleEG5 + process.HLTDoEgammaClusterShapeSequence + process.hltSingleEle6CaloIdTTrlIdVLClusterShapeFilter + process.HLTDoEGammaHESequence + process.hltSingleEle6CaloIdTHEFilter + process.HLTDoEGammaPixelSequence + process.hltSingleEle6CaloIdTPixelMatchFilter )
process.HLTPixelMatchElectronL1SeededTrackingSequence = cms.Sequence( process.hltCkfL1SeededTrackCandidates + process.hltCtfL1SeededWithMaterialTracks + process.hltPixelMatchElectronsL1Seeded )
process.HLTDoElectronDetaDphiSequence = cms.Sequence( process.hltElectronL1SeededDetaDphi )
process.HLTSingleEle6CaloIdNoneSequence = cms.Sequence( process.HLTDoEGammaStartupSequence + process.hltEGRegionalL1SingleEG5PA + process.hltSingleEG6EtFilterL1SingleEG5 + process.HLTDoEgammaClusterShapeSequence + process.hltSingleEle6CaloIdNoneTrlIdVLClusterShapeFilter + process.HLTDoEGammaHESequence + process.hltSingleEle6CaloIdNoneHEFilter + process.HLTDoEGammaPixelSequence + process.hltSingleEle6CaloIdNonePixelMatchFilter )
process.HLTSingleEle8CaloIdNoneSequence = cms.Sequence( process.HLTDoEGammaStartupSequence + process.hltEGRegionalL1SingleEG7 + process.hltSingleEG8EtFilterL1SingleEG7 + process.HLTDoEgammaClusterShapeSequence + process.hltSingleEle8CaloIdNoneTrlIdVLClusterShapeFilter + process.HLTDoEGammaHESequence + process.hltSingleEle8CaloIdNoneHEFilter + process.HLTDoEGammaPixelSequence + process.hltSingleEle8CaloIdNonePixelMatchFilter )
process.HLTDoubleEG5DoubleEle6CaloIdTSequence = cms.Sequence( process.HLTDoEGammaStartupSequence + process.hltEGRegionalL1DoubleEG5 + process.hltDoubleEG5DoubleEle6EtFilter + process.HLTDoEgammaClusterShapeSequence + process.hltDoubleEG5DoubleEle6CaloIdTTrlIdVLClusterShapeFilter + process.HLTDoEGammaHESequence + process.hltDoubleEG5DoubleEle6CaloIdTHEFilter + process.HLTDoEGammaPixelSequence + process.hltDoubleEG5DoubleEle6CaloIdTPixelMatchFilter )
process.HLTDoubleEle6CaloIdTSequence = cms.Sequence( process.HLTDoEGammaStartupSequence + process.hltEGRegionalL1SingleEG5PA + process.hltDoubleEG6EtFilterL1SingleEG5 + process.HLTDoEgammaClusterShapeSequence + process.hltDoubleEle6CaloIdTTrlIdVLClusterShapeFilter + process.HLTDoEGammaHESequence + process.hltDoubleEle6CaloIdTHEFilter + process.HLTDoEGammaPixelSequence + process.hltDoubleEle6CaloIdTPixelMatchFilter )
process.HLTDoubleEle8CaloIdTSequence = cms.Sequence( process.HLTDoEGammaStartupSequence + process.hltEGRegionalL1SingleEG7 + process.hltDoubleEG8EtFilterL1SingleEG7 + process.HLTDoEgammaClusterShapeSequence + process.hltDoubleEle8CaloIdTTrlIdVLClusterShapeFilter + process.HLTDoEGammaHESequence + process.hltDoubleEle8CaloIdTHEFilter + process.HLTDoEGammaPixelSequence + process.hltDoubleEle8CaloIdTPixelMatchFilter )
process.HLTRecopixelvertexingForHighMultPASequence = cms.Sequence( process.hltPAPixelTracksForHighMult + process.hltPAPixelVerticesForHighMult )
process.HLTIterativeTrackingIteration0ForPA = cms.Sequence( process.hltPAPixelTracksForHighPt + process.hltPAPixelSeedsFromPixelTracks + process.hltPACkfTrackCandidates + process.hltPACtfWithMaterialTracks + process.hltPATrackSelectionHighPurity + process.hltPATrackRefsForJetsIter0 + process.hltPAAntiKT5TrackJetsIter0 + process.hltPATrackAndTauJetsIter0 )
process.HLTIterativeTrackingIteration1ForPA = cms.Sequence( process.hltPAIter1ClustersRefRemoval + process.hltPAIter1SiStripClusters + process.hltPAIter1PixelSeeds + process.hltPAIter1CkfTrackCandidates + process.hltPAIter1CtfWithMaterialTracks + process.hltPAIter1TrackSelectionHighPurityLoose + process.hltPAIter1TrackSelectionHighPurityTight + process.hltPAIter1TrackSelectionHighPurity + process.hltPAIter1Merged + process.hltPATrackRefsForJetsIter1 + process.hltPAAntiKT5TrackJetsIter1 + process.hltPATrackAndTauJetsIter1 )
process.HLTIterativeTrackingIteration2ForPA = cms.Sequence( process.hltPAIter2ClustersRefRemoval + process.hltPAIter2SiStripClusters + process.hltPAIter2PixelSeeds + process.hltPAIter2CkfTrackCandidates + process.hltPAIter2CtfWithMaterialTracks + process.hltPAIter2TrackSelectionHighPurity + process.hltPAIter2Merged + process.hltPATrackRefsForJetsIter2 + process.hltPAAntiKT5TrackJetsIter2 + process.hltPATrackAndTauJetsIter2 )
process.HLTIterativeTrackingIteration3ForPA = cms.Sequence( process.hltPAIter3ClustersRefRemoval + process.hltPAIter3SiStripClusters + process.hltPAIter3MixedSeeds + process.hltPAIter3CkfTrackCandidates + process.hltPAIter3CtfWithMaterialTracks + process.hltPAIter3TrackSelectionHighPurityLoose + process.hltPAIter3TrackSelectionHighPurityTight + process.hltPAIter3TrackSelectionHighPurity + process.hltPAIter3Merged + process.hltPATrackRefsForJetsIter3 + process.hltPAAntiKT5TrackJetsIter3 + process.hltPATrackAndTauJetsIter3 )
process.HLTIterativeTrackingIteration4ForPA = cms.Sequence( process.hltPAIter4ClustersRefRemoval + process.hltPAIter4SiStripClusters + process.hltPAIter4PixelLessSeeds + process.hltPAIter4CkfTrackCandidates + process.hltPAIter4CtfWithMaterialTracks + process.hltPAIter4TrackSelectionHighPurity + process.hltPAIter4Merged )
process.HLTIterativeTrackingForPA = cms.Sequence( process.HLTIterativeTrackingIteration0ForPA + process.HLTIterativeTrackingIteration1ForPA + process.HLTIterativeTrackingIteration2ForPA + process.HLTIterativeTrackingIteration3ForPA + process.HLTIterativeTrackingIteration4ForPA )
process.HLTPixelTrackingForPAMinBiasSequence = cms.Sequence( process.hltPAPixelTracksForMinBias )
process.HLTPAUpcFullTrackRecoSequence = cms.Sequence( process.HLTDoLocalStripSequence + process.hltPAUpcTrackSeeds + process.hltPAUpcCkfTrackCandidates + process.hltPAUpcCtfTracks + process.hltPAUpcCtfTrackCands )
process.HLTMuTrackUpcOniaPixelRecoSequence = cms.Sequence( process.HLTDoLocalPixelSequence + process.hltPixelTracks + process.hltMuTrackPixelTrackSelectorUpcOnia + process.hltMuTrackPixelTrackCandsUpcOnia )
process.HLTMuTrackUpcOniaTrackRecoSequence = cms.Sequence( process.HLTDoLocalStripSequence + process.hltMuTrackTrackSeedsUpcOnia + process.hltMuTrackCkfTrackCandidatesUpcOnia + process.hltMuTrackCtfTracksUpcOnia + process.hltMuTrackCtfTrackCandsUpcOnia )
process.HLTMuTkMuUpcOniaTkMuRecoSequence = cms.Sequence( process.hltMuTkMuMuonLinksUpcOnia + process.hltMuTkMuMuonsUpcOnia + process.hltMuTkMuTrackerMuonCandsUpcOnia )

process.HLTriggerFirstPath = cms.Path( process.hltGetConditions + process.hltGetRaw + process.hltBoolFalse )
process.HLT_Activity_Ecal_SC7_v13 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPreActivityEcalSC7 + process.HLTEcalActivitySequence + process.hltEgammaSelectEcalSuperClustersActivityFilterSC7 + process.HLTEndSequence )
process.HLT_Mu15_eta2p1_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu7 + process.hltPreMu15eta2p1 + process.hltL1fL1sMu7L1FilteredEta2p1Filtered0 + process.HLTL2muonrecoSequence + process.hltL2fL1sMu7L1fEta2p1L2FilteredEta2p1Filtered7 + process.HLTL3muonrecoSequence + process.hltL3fL1sMu7L1fEta2p1L2fEta2p1f7L3FilteredEta2p1Filtered15 + process.HLTEndSequence )
process.HLT_Ele22_CaloIdL_CaloIsoVL_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG12 + process.hltPreEle22CaloIdLCaloIsoVL + process.HLTEle22CaloIdLCaloIsoVLSequence + process.HLTEndSequence )
process.HLT_BeamGas_HF_Beam1_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BeamGasHfBptxPlusPostQuiet + process.hltPreBeamGasHFBeam1 + process.hltHcalDigis + process.hltHfreco + process.hltHFAsymmetryFilterTight + process.HLTEndSequence )
process.HLT_BeamGas_HF_Beam2_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BeamGasHfBptxMinusPostQuiet + process.hltPreBeamGasHFBeam2 + process.hltHcalDigis + process.hltHfreco + process.hltHFAsymmetryFilterTight + process.HLTEndSequence )
process.HLT_BeamHalo_v13 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BeamHalo + process.hltPreBeamHalo + process.HLTDoLocalPixelClustersSequence + process.hltPixelActivityFilterForHalo + process.HLTDoLocalStripSequence + process.hltTrackerHaloFilter + process.HLTEndSequence )
process.HLT_PAHcalUTCA_v1 = cms.Path( process.HLTPABeginSequenceUTCA + process.hltPrePAHcalUTCA + process.HLTEndSequence )
process.HLT_PAHcalPhiSym_v1 = cms.Path( process.HLTBeginSequenceNZS + process.hltL1sPAHcalPhiSym + process.hltPrePAHcalPhiSym + process.HLTEndSequence )
process.HLT_PAHcalNZS_v1 = cms.Path( process.HLTBeginSequenceNZS + process.hltL1sPAHcalNZS + process.hltPrePAHcalNZS + process.HLTEndSequence )
process.HLT_GlobalRunHPDNoise_v8 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet20CentralNoBPTXNoHalo + process.hltPreGlobalRunHPDNoise + process.HLTEndSequence )
process.HLT_Physics_v5 = cms.Path( process.HLTBeginSequence + process.hltPrePhysics + process.HLTEndSequence )
process.DST_Physics_v5 = cms.Path( process.HLTBeginSequence + process.hltPreDSTPhysics + process.HLTEndSequence )
process.HLT_DTCalibration_v2 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreDTCalibration + process.hltDTCalibrationRaw + process.HLTEndSequence )
process.HLT_EcalCalibration_v3 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreEcalCalibration + process.hltEcalCalibrationRaw + process.HLTEndSequence )
process.HLT_HcalCalibration_v3 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreHcalCalibration + process.hltHcalCalibTypeFilter + process.hltHcalCalibrationRaw + process.HLTEndSequence )
process.HLT_TrackerCalibration_v3 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreTrackerCalibration + process.hltLaserAlignmentEventFilter + process.hltTrackerCalibrationRaw + process.HLTEndSequence )
process.HLT_L1SingleMuOpen_AntiBPTX_v7 = cms.Path( process.HLTBeginSequenceAntiBPTX + process.hltL1sL1SingleMuOpen + process.hltPreL1SingleMuOpenAntiBPTX + process.hltL1MuOpenL1Filtered0 + process.HLTEndSequence )
process.HLT_L1TrackerCosmics_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sTrackerCosmics + process.hltPreL1TrackerCosmics + process.hltTrackerCosmicsPattern + process.HLTEndSequence )
process.AlCa_PAEcalPi0EBonly_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaPAEcalPi0Eta + process.hltPreAlCaPAEcalPi0EBonly + process.HLTDoRegionalPi0EtaSequence + process.hltSimple3x3Clusters + process.hltAlCaPi0RecHitsFilterEBonly + process.hltAlCaPi0EBUncalibrator + process.HLTEndSequence )
process.AlCa_PAEcalPi0EEonly_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaPAEcalPi0Eta + process.hltPreAlCaPAEcalPi0EEonly + process.HLTDoRegionalPi0EtaSequence + process.hltSimple3x3Clusters + process.hltAlCaPi0RecHitsFilterEEonly + process.hltAlCaPi0EEUncalibrator + process.HLTEndSequence )
process.AlCa_PAEcalEtaEBonly_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaPAEcalPi0Eta + process.hltPreAlCaPAEcalEtaEBonly + process.HLTDoRegionalPi0EtaSequence + process.hltSimple3x3Clusters + process.hltAlCaEtaRecHitsFilterEBonly + process.hltAlCaEtaEBUncalibrator + process.HLTEndSequence )
process.AlCa_PAEcalEtaEEonly_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaPAEcalPi0Eta + process.hltPreAlCaPAEcalEtaEEonly + process.HLTDoRegionalPi0EtaSequence + process.hltSimple3x3Clusters + process.hltAlCaEtaRecHitsFilterEEonly + process.hltAlCaEtaEEUncalibrator + process.HLTEndSequence )
process.AlCa_EcalPhiSym_v13 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPreAlCaEcalPhiSym + process.hltEcalRawToRecHitFacility + process.hltESRawToRecHitFacility + process.hltEcalRegionalRestFEDs + process.hltEcalRecHitAll + process.hltAlCaPhiSymStream + process.hltAlCaPhiSymUncalibrator + process.HLTEndSequence )
process.AlCa_RPCMuonNoTriggers_v9 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaRPC + process.hltPreAlCaRPCMuonNoTriggers + process.hltRPCMuonNoTriggersL1Filtered0 + process.HLTMuonLocalRecoSequence + process.HLTEndSequence )
process.AlCa_RPCMuonNoHits_v9 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaRPC + process.hltPreAlCaRPCMuonNoHits + process.HLTMuonLocalRecoSequence + process.hltRPCPointProducer + process.hltRPCFilter + process.HLTEndSequence )
process.AlCa_RPCMuonNormalisation_v9 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaRPC + process.hltPreAlCaRPCMuonNormalisation + process.hltRPCMuonNormaL1Filtered0 + process.HLTMuonLocalRecoSequence + process.HLTEndSequence )
process.AlCa_LumiPixels_v8 = cms.Path( process.HLTBeginSequence + process.hltL1sL1AlwaysTrue + process.hltPreAlCaLumiPixels + process.hltFEDSelectorLumiPixels + process.HLTEndSequence )
process.AlCa_LumiPixels_ZeroBias_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPreAlCaLumiPixelsZeroBias + process.hltFEDSelectorLumiPixels + process.HLTEndSequence )
process.AlCa_LumiPixels_Random_v1 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreAlCaLumiPixelsRandom + process.hltFEDSelectorLumiPixels + process.HLTEndSequence )
process.HLT_PAL1SingleJet16_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16BptxAND + process.hltPrePAL1SingleJet16 + process.HLTEndSequence )
process.HLT_PAL1SingleJet36_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36 + process.hltPrePAL1SingleJet36 + process.HLTEndSequence )
process.HLT_PASingleForJet15_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1ZeroBias + process.hltPrePASingleForJet15 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet15 + process.HLTEndSequence )
process.HLT_PASingleForJet25_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleForJet16 + process.hltPrePASingleForJet25 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet25 + process.HLTEndSequence )
process.HLT_PAJet20_NoJetID_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16BptxAND + process.hltPrePAJet20NoJetID + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet20RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAJet40_NoJetID_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16BptxAND + process.hltPrePAJet40NoJetID + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet40RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAJet60_NoJetID_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36 + process.hltPrePAJet60NoJetID + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet60RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAJet80_NoJetID_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36 + process.hltPrePAJet80NoJetID + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet80RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAJet100_NoJetID_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36 + process.hltPrePAJet100NoJetID + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet100RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAJet120_NoJetID_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36 + process.hltPrePAJet120NoJetID + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet120RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAForJet20Eta2_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16BptxANDinForJet20Eta2 + process.hltPrePAForJet20Eta2 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet20Eta2 + process.HLTEndSequence )
process.HLT_PAForJet40Eta2_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet40Eta2 + process.hltPrePAForJet40Eta2 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet40Eta2 + process.HLTEndSequence )
process.HLT_PAForJet60Eta2_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet60Eta2 + process.hltPrePAForJet60Eta2 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet60Eta2 + process.HLTEndSequence )
process.HLT_PAForJet80Eta2_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet80Eta2 + process.hltPrePAForJet80Eta2 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet80Eta2 + process.HLTEndSequence )
process.HLT_PAForJet100Eta2_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet100Eta2 + process.hltPrePAForJet100Eta2 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet100Eta2 + process.HLTEndSequence )
process.HLT_PAForJet20Eta3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16BptxANDinForJet20Eta3 + process.hltPrePAForJet20Eta3 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet20Eta3 + process.HLTEndSequence )
process.HLT_PAForJet40Eta3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet40Eta3 + process.hltPrePAForJet40Eta3 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet40Eta3 + process.HLTEndSequence )
process.HLT_PAForJet60Eta3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet60Eta3 + process.hltPrePAForJet60Eta3 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet60Eta3 + process.HLTEndSequence )
process.HLT_PAForJet80Eta3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet80Eta3 + process.hltPrePAForJet80Eta3 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet80Eta3 + process.HLTEndSequence )
process.HLT_PAForJet100Eta3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inForJet100Eta3 + process.hltPrePAForJet100Eta3 + process.HLTRecoJetSequenceAK5Corrected + process.hltSingleForJet100Eta3 + process.HLTEndSequence )
process.HLT_PATripleJet20_20_20_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16BptxANDinTripleJet202020 + process.hltPrePATripleJet202020 + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet20RegionalNoJetIDinTripleJet202020 + process.hltSecondJet20RegionalNoJetID + process.hltThirdJet20RegionalNoJetID + process.HLTEndSequence )
process.HLT_PATripleJet40_20_20_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inTripleJet402020 + process.hltPrePATripleJet402020 + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet40RegionalNoJetID + process.hltSecondJet20RegionalNoJetID + process.hltThirdJet20RegionalNoJetID + process.HLTEndSequence )
process.HLT_PATripleJet60_20_20_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inTripleJet602020 + process.hltPrePATripleJet602020 + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet60RegionalNoJetID + process.hltSecondJet20RegionalNoJetID + process.hltThirdJet20RegionalNoJetID + process.HLTEndSequence )
process.HLT_PATripleJet80_20_20_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inTripleJet802020 + process.hltPrePATripleJet802020 + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet80RegionalNoJetID + process.hltSecondJet20RegionalNoJetID + process.hltThirdJet20RegionalNoJetID + process.HLTEndSequence )
process.HLT_PATripleJet100_20_20_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36inTripleJet1002020 + process.hltPrePATripleJet1002020 + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet100RegionalNoJetIDinTripleJet1002020 + process.hltSecondJet20RegionalNoJetID + process.hltThirdJet20RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAJet40ETM30_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16BptxANDAndETM30 + process.hltPrePAJet40ETM30 + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet40RegionalNoJetID + process.HLTEndSequence )
process.HLT_PAJet60ETM30_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36AndETM30 + process.hltPrePAJet60ETM30 + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet60RegionalNoJetIDinJet60ETM30 + process.HLTEndSequence )
process.HLT_PAL1DoubleMu0_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleMu0 + process.hltPrePAL1DoubleMu0 + process.hltL1fL1sL1DoubleMu0L1f0 + process.HLTEndSequence )
process.HLT_PADimuon0_NoVertexing_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleMu0erHighQ + process.hltPrePADimuon0NoVertexing + process.hltDimuonL1Filtered0 + process.HLTL2muonrecoSequence + process.hltDimuonL2PreFiltered0 + process.HLTEndSequence )
process.HLT_PAL1DoubleMu0_HighQ_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleMuOpenBptxAnd + process.hltL1fL1sPAL1DoubleMu0HighQL1FilteredHighQ + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAL1DoubleMu0HighQ + process.HLTEndSequence )
process.HLT_PAL1DoubleMuOpen_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleMuOpenBptxAnd + process.hltL1fL1sPAL1DoubleMuOpenL1Filtered0 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAL1DoubleMuOpen + process.HLTEndSequence )
process.HLT_PAL2DoubleMu3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleMuOpenBptxAnd + process.hltL1fL1sPAL2DoubleMu3L1Filtered0 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAL2DoubleMu3 + process.HLTL2muonrecoSequence + process.hltL2fL1sPAL2DoubleMu3L2Filtered3 + process.HLTEndSequence )
process.HLT_PAMu3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMu3 + process.hltL1fL1sMu3L1Filtered0 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAMu3 + process.HLTL2muonrecoSequence + process.hltL2fL1sMu3L2Filtered3 + process.HLTL3muonrecoSequence + process.hltL3fL2sMu3L3Filtered3 + process.HLTEndSequence )
process.HLT_PAMu7_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMu7 + process.hltL1fL1sMu7L1Filtered0 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAMu7 + process.HLTL2muonrecoSequence + process.hltL2fL1sMu7L2Filtered5 + process.HLTL3muonrecoSequence + process.hltL3fL2sMu7L3Filtered7 + process.HLTEndSequence )
process.HLT_PAMu12_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMu12 + process.hltL1fL1sMu12L1Filtered0 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAMu12 + process.HLTL2muonrecoSequence + process.hltL2fL1sMu12L2Filtered10 + process.HLTL3muonrecoSequence + process.hltL3fL2sMu12L3Filtered12 + process.HLTEndSequence )
process.HLT_PABTagMu_Jet20_Mu4_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1Mu3JetC16WdEtaPhi2 + process.hltPrePABTagMuJet20Mu4 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.HLTRecoJetSequenceAK5L1FastJetCorrected + process.hltBJet20L1FastJetCentralBPH + process.HLTBTagMuJet20L1FastJetSequenceL25BPH + process.hltBSoftMuonJet20L1FastJetL25FilterByDRBPH + process.HLTBTagMuJet20L1FastJetMu5SelSequenceL3BPH + process.hltBSoftMuonJet20L1FastJetMu5L3FilterByDRBPH + process.HLTEndSequence )
process.HLT_PAMu3PFJet20_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMu3Jet16 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAMu3PFJet20 + process.HLTL2muonrecoSequence + process.hltPAMu3PFJet20L2Filtered3 + process.HLTL3muonrecoSequence + process.hltPAMu3PFJet20L3Filter3 + process.HLTPFL1FastL2L3ReconstructionSequence + process.hltPFJetsL1Matched + process.hltPAMu3PFJet20 + process.HLTEndSequence )
process.HLT_PAMu3PFJet40_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltPAL1sL1SingleMu3Jet36 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAMu3PFJet40 + process.HLTL2muonrecoSequence + process.hltPAMu3PFJet40L2Filtered3 + process.HLTL3muonrecoSequence + process.hltPAMu3PFJet40L3Filter3 + process.HLTPFL1FastL2L3ReconstructionSequence + process.hltPFJetsL1Matched + process.hltPAMu3PFJet40 + process.HLTEndSequence )
process.HLT_PAMu7PFJet20_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMu7Jet16 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAMu7PFJet20 + process.HLTL2muonrecoSequence + process.hltPAMu7PFJet20L2Filtered5 + process.HLTL3muonrecoSequence + process.hltPAMu7PFJet20L3Filter3 + process.HLTPFL1FastL2L3ReconstructionSequence + process.hltPFJetsL1Matched + process.hltPAMu7PFJet20 + process.HLTEndSequence )
process.HLT_PAPhoton10_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton10NoCaloIdVL + process.HLTPhoton10NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton15_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton15NoCaloIdVL + process.HLTPhoton15NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton20_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton20NoCaloIdVL + process.HLTPhoton20NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton30_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG12 + process.hltPrePAPhoton30NoCaloIdVL + process.HLTPhoton30NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton40_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG20 + process.hltPrePAPhoton40NoCaloIdVL + process.HLTPhoton40NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton60_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG24 + process.hltPrePAPhoton60NoCaloIdVL + process.HLTPhoton60NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton10_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton10TightCaloIdVL + process.HLTPAPhoton10TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton15_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton15TightCaloIdVL + process.HLTPAPhoton15TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton20_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton20TightCaloIdVL + process.HLTPAPhoton20TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton30_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG12 + process.hltPrePAPhoton30TightCaloIdVL + process.HLTPAPhoton30TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton40_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG20 + process.hltPrePAPhoton40TightCaloIdVL + process.HLTPAPhoton40TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton10_TightCaloIdVL_Iso50_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton10TightCaloIdVLIso50 + process.HLTPAPhoton10TightCaloIdVLIso50Sequence + process.HLTEndSequence )
process.HLT_PAPhoton15_TightCaloIdVL_Iso50_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton15TightCaloIdVLIso50 + process.HLTPAPhoton15TightCaloIdVLIso50Sequence + process.HLTEndSequence )
process.HLT_PAPhoton20_TightCaloIdVL_Iso50_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG5BptxAND + process.hltPrePAPhoton20TightCaloIdVLIso50 + process.HLTPAPhoton20TightCaloIdVLIso50Sequence + process.HLTEndSequence )
process.HLT_PAPhoton30_TightCaloIdVL_Iso50_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleEG12 + process.hltPrePAPhoton30TightCaloIdVLIso50 + process.HLTPAPhoton30TightCaloIdVLIso50Sequence + process.HLTEndSequence )
process.HLT_PAPhoton10_Photon10_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton10Photon10NoCaloIdVL + process.HLTDoublePhoton10And10NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton15_Photon10_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton15Photon10NoCaloIdVL + process.HLTDoublePhoton15And10NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton20_Photon15_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton20Photon15NoCaloIdVL + process.HLTDoublePhoton20And15NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton20_Photon20_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton20Photon20NoCaloIdVL + process.HLTDoublePhoton20And20NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton30_Photon30_NoCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton30Photon30NoCaloIdVL + process.HLTDoublePhoton30And30NoCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton10_Photon10_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton10Photon10TightCaloIdVL + process.HLTPADoublePhoton10And10TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton10Photon10TightCaloIdVLIso50 + process.HLTPADoublePhoton10And10TightCaloIdVLIso50Sequence + process.HLTEndSequence )
process.HLT_PAPhoton15_Photon10_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton15Photon10TightCaloIdVL + process.HLTPADoublePhoton15And10TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PAPhoton20_Photon15_TightCaloIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAPhoton20Photon15TightCaloIdVL + process.HLTPADoublePhoton20And15TightCaloIdVLSequence + process.HLTEndSequence )
process.HLT_PASingleEle6_CaloIdT_TrkIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPrePASingleEle6CaloIdTTrkIdVL + process.HLTSingleEle6CaloIdTSequence + process.HLTPixelMatchElectronL1SeededTrackingSequence + process.hltEle6CaloIdTOneOEMinusOneOPSingleFilter + process.HLTDoElectronDetaDphiSequence + process.hltEle6CaloIdTTrkIdVLDetaSingleFilter + process.hltEle6CaloIdTTrkIdVLDphiSingleFilter + process.HLTEndSequence )
process.HLT_PASingleEle6_CaloIdNone_TrkIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPrePASingleEle6CaloIdNoneTrkIdVL + process.HLTSingleEle6CaloIdNoneSequence + process.HLTPixelMatchElectronL1SeededTrackingSequence + process.hltEle6CaloIdNoneOneOEMinusOneOPSingleFilter + process.HLTDoElectronDetaDphiSequence + process.hltEle6CaloIdNoneTrkIdVLDetaSingleFilter + process.hltEle6CaloIdNoneTrkIdVLDphiSingleFilter + process.HLTEndSequence )
process.HLT_PASingleEle8_CaloIdNone_TrkIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7 + process.hltPrePASingleEle8CaloIdNoneTrkIdVL + process.HLTSingleEle8CaloIdNoneSequence + process.HLTPixelMatchElectronL1SeededTrackingSequence + process.hltEle8CaloIdNoneOneOEMinusOneOPSingleFilter + process.HLTDoElectronDetaDphiSequence + process.hltEle8CaloIdNoneTrkIdVLDetaSingleFilter + process.hltEle8CaloIdNoneTrkIdVLDphiSingleFilter + process.HLTEndSequence )
process.HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5 + process.hltPrePAL1DoubleEG5DoubleEle6CaloIdTTrkIdVL + process.HLTDoubleEG5DoubleEle6CaloIdTSequence + process.HLTPixelMatchElectronL1SeededTrackingSequence + process.hltDoubleEG5DoubleEle6CaloIdTOneOEMinusOneOPDoubleFilter + process.HLTDoElectronDetaDphiSequence + process.hltDoubleEG5DoubleEle6CaloIdTTrkIdVLDetaDoubleFilter + process.hltDoubleEG5DoubleEle6CaloIdTTrkIdVLDphiDoubleFilter + process.HLTEndSequence )
process.HLT_PADoubleEle6_CaloIdT_TrkIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPrePADoubleEle6CaloIdTTrkIdVL + process.HLTDoubleEle6CaloIdTSequence + process.HLTPixelMatchElectronL1SeededTrackingSequence + process.hltEle6CaloIdTOneOEMinusOneOPDoubleFilter + process.HLTDoElectronDetaDphiSequence + process.hltEle6CaloIdTTrkIdVLDetaDoubleFilter + process.hltEle6CaloIdTTrkIdVLDphiDoubleFilter + process.HLTEndSequence )
process.HLT_PADoubleEle8_CaloIdT_TrkIdVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7 + process.hltPrePADoubleEle8CaloIdTTrkIdVL + process.HLTDoubleEle8CaloIdTSequence + process.HLTPixelMatchElectronL1SeededTrackingSequence + process.hltEle8CaloIdTOneOEMinusOneOPDoubleFilter + process.HLTDoElectronDetaDphiSequence + process.hltEle8CaloIdTTrkIdVLDetaDoubleFilter + process.hltEle8CaloIdTTrkIdVLDphiDoubleFilter + process.HLTEndSequence )
process.HLT_PAPixelTracks_Multiplicity100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePAPixelTracksMultiplicity100 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult100 + process.HLTEndSequence )
process.HLT_PAPixelTracks_Multiplicity130_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePAPixelTracksMultiplicity130 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult130 + process.HLTEndSequence )
process.HLT_PAPixelTracks_Multiplicity160_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT40 + process.hltPrePAPixelTracksMultiplicity160 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult160 + process.HLTEndSequence )
process.HLT_PAPixelTracks_Multiplicity190_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT40 + process.hltPrePAPixelTracksMultiplicity190 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult190 + process.HLTEndSequence )
process.HLT_PAPixelTracks_Multiplicity220_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT60 + process.hltPrePAPixelTracksMultiplicity220 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult220 + process.HLTEndSequence )
process.HLT_PAPixelTrackMultiplicity100_FullTrack12_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePAPixelTrackMultiplicity100FullTrack12 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult100 + process.HLTRecoJetSequencePrePF + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack12 + process.HLTEndSequence )
process.HLT_PAPixelTrackMultiplicity130_FullTrack12_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePAPixelTrackMultiplicity130FullTrack12 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult130 + process.HLTRecoJetSequencePrePF + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack12 + process.HLTEndSequence )
process.HLT_PAPixelTrackMultiplicity160_FullTrack12_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sETT40 + process.hltPrePAPixelTrackMultiplicity160FullTrack12 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult160 + process.HLTRecoJetSequencePrePF + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack12 + process.HLTEndSequence )
process.HLT_PAFullTrack12_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet12BptxAND + process.hltPrePAFullTrack12 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTRecoJetSequencePrePF + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack12 + process.HLTEndSequence )
process.HLT_PAFullTrack20_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet16BptxAND + process.hltPrePAFullTrack20 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTRecoJetSequencePrePF + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack20 + process.HLTEndSequence )
process.HLT_PAFullTrack30_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet16BptxAND + process.hltPrePAFullTrack30 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTRecoJetSequencePrePF + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack30 + process.HLTEndSequence )
process.HLT_PAFullTrack50_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet36 + process.hltPrePAFullTrack50 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTRecoJetSequencePrePF + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack50 + process.HLTEndSequence )
process.HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v2 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet36 + process.hltPrePAPixelTrackMultiplicity140Jet80NoJetID + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.HLTRegionalTowerMakerForJetsSequence + process.hltAntiKT5CaloJetsRegional + process.hltCaloJetL1MatchedRegional + process.hltCaloJetCorrectedRegionalNoJetID + process.hltSingleJet80RegionalNoJetID + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult140 + process.HLTEndSequence )
process.HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleMuOpenBptxAnd + process.hltL1fL1sPAL2DoubleMu3L1Filtered0 + process.HLTDoLocalHfSequence + process.hltHcalPM1Tower3GeVFilter + process.hltPrePAPixelTrackMultiplicity100L2DoubleMu3 + process.HLTL2muonrecoSequence + process.hltL2fL1sPAL2DoubleMu3L2Filtered3 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult100 + process.HLTEndSequence )
process.HLT_PPPixelTracks_Multiplicity55_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePPPixelTracksMultiplicity55 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult55 + process.HLTEndSequence )
process.HLT_PPPixelTracks_Multiplicity70_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePPPixelTracksMultiplicity70 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult70 + process.HLTEndSequence )
process.HLT_PPPixelTracks_Multiplicity85_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePPPixelTracksMultiplicity85 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult85 + process.HLTEndSequence )
process.HLT_PPPixelTrackMultiplicity55_FullTrack12_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePPPixelTrackMultiplicity55FullTrack12 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult55 + process.HLTRecoJetSequencePrePF + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack12 + process.HLTEndSequence )
process.HLT_PPPixelTrackMultiplicity70_FullTrack12_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePPPixelTrackMultiplicity70FullTrack12 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + process.hltPAGoodPixelTracksForHighMult + process.hltPAPixelCandsForHighMult + process.hlt1PAHighMult70 + process.HLTRecoJetSequencePrePF + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingForPA + process.hltPAGoodFullTracks + process.hltPAFullCands + process.hlt1PAFullTrack12 + process.HLTEndSequence )
process.HLT_PPL1DoubleJetC36_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleJetC36 + process.hltPrePPL1DoubleJetC36 + process.HLTEndSequence )
process.HLT_PATech35_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sPATech35 + process.hltPrePATech35 + process.HLTEndSequence )
process.HLT_PATech35_HFSumET100_v3 = cms.Path( process.HLTBeginSequence + process.hltL1sPATech35 + process.hltPrePATech35HFSumET100 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.hltPAMetForHf + process.hltGlobalSumETHfFilter100 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + ~process.hlt1PAVertexFilter + process.hltPAPixelCandsForHFSumET + process.hlt1PAHighMult3ForHFSumET + process.HLTEndSequence )
process.HLT_PAHFSumET100_v3 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePAHFSumET100 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.hltPAMetForHf + process.hltGlobalSumETHfFilter100 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + ~process.hlt1PAVertexFilter + process.hltPAPixelCandsForHFSumET + process.hlt1PAHighMult3ForHFSumET + process.HLTEndSequence )
process.HLT_PAHFSumET140_v3 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePAHFSumET140 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.hltPAMetForHf + process.hltGlobalSumETHfFilter140 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + ~process.hlt1PAVertexFilter + process.hltPAPixelCandsForHFSumET + process.hlt1PAHighMult3ForHFSumET + process.HLTEndSequence )
process.HLT_PAHFSumET170_v3 = cms.Path( process.HLTBeginSequence + process.hltL1sETT20BptxAND + process.hltPrePAHFSumET170 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.hltPAMetForHf + process.hltGlobalSumETHfFilter170 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + ~process.hlt1PAVertexFilter + process.hltPAPixelCandsForHFSumET + process.hlt1PAHighMult3ForHFSumET + process.HLTEndSequence )
process.HLT_PAHFSumET210_v3 = cms.Path( process.HLTBeginSequence + process.hltL1sETT40 + process.hltPrePAHFSumET210 + process.HLTDoLocalHfSequence + process.hltHcalPM2Tower3GeVFilter + process.hltPAMetForHf + process.hltGlobalSumETHfFilter210 + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingForHighMultPASequence + ~process.hlt1PAVertexFilter + process.hltPAPixelCandsForHFSumET + process.hlt1PAHighMult3ForHFSumET + process.HLTEndSequence )
process.HLT_PARomanPots_Tech52_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sRomanPotsTech52 + process.hltPrePARomanPotsTech52 + process.HLTEndSequence )
process.HLT_PAL1Tech53_MB_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Tech53MB + process.hltPrePAL1Tech53MB + process.HLTEndSequence )
process.HLT_PAL1Tech53_MB_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Tech53MB + process.hltPrePAL1Tech53MBSingleTrack + process.HLTDoLocalPixelSequence + process.HLTPixelTrackingForPAMinBiasSequence + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + process.HLTEndSequence )
process.HLT_PAL1Tech54_ZeroBias_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Tech54ZeroBias + process.hltPrePAL1Tech54ZeroBias + process.HLTEndSequence )
process.HLT_PAT1minbias_Tech55_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sT1minbiasTech55 + process.hltPrePAT1minbiasTech55 + process.HLTEndSequence )
process.HLT_PAL1Tech_HBHEHO_totalOR_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sTechTrigHCALNoise + process.hltPrePAL1TechHBHEHOtotalOR + process.HLTEndSequence )
process.HLT_PAL1Tech63_CASTORHaloMuon_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Tech63CASTORHaloMuon + process.hltPrePAL1Tech63CASTORHaloMuon + process.HLTEndSequence )
process.HLT_PACastorEmTotemLowMultiplicity_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1CastorEmTotemLowMultiplicity + process.hltPrePACastorEmTotemLowMultiplicity + process.HLTDoLocalPixelSequence + process.HLTPixelTrackingForPAMinBiasSequence + process.hltPAPixelCandsForMinBias + ~process.hltPixelTrackMultVetoFilterCastor + process.HLTEndSequence )
process.HLT_PACastorEmNotHfCoincidencePm_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1CastorEmNotHfCoincidencePm + process.hltPrePACastorEmNotHfCoincidencePm + process.HLTDoLocalPixelSequence + process.HLTPixelTrackingForPAMinBiasSequence + process.hltPAPixelCandsForMinBias + process.hltPixelTrackFilterCastorHfMin + ~process.hltPixelTrackMultVetoFilterCastorHfMax + process.HLTEndSequence )
process.HLT_PACastorEmNotHfSingleChannel_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1CastorEmNotHfSingleChannel + process.hltPrePACastorEmNotHfSingleChannel + process.HLTDoLocalPixelSequence + process.HLTPixelTrackingForPAMinBiasSequence + process.hltPAPixelCandsForMinBias + process.hltPixelTrackFilterCastorHfMin + ~process.hltPixelTrackMultVetoFilterCastorHfMax + process.HLTEndSequence )
process.HLT_PAL1CastorTotalTotemLowMultiplicity_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1CastorTotalTotemLowMultiplicity + process.hltPrePAL1CastorTotalTotemLowMultiplicity + process.HLTEndSequence )
process.HLT_PAMinBiasHF_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sPAMinBiasHFBptxAND + process.hltPrePAMinBiasHF + process.HLTEndSequence )
process.HLT_PAMinBiasHF_OR_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sPAMinBiasHFORBptxAND + process.hltPrePAMinBiasHFOR + process.HLTEndSequence )
process.HLT_PAMinBiasBHC_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sBscMinBiasThreshold1BptxAND + process.hltPrePAMinBiasBHC + process.HLTEndSequence )
process.HLT_PAMinBiasBHC_OR_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sPAMinBiasBscBptxAND + process.hltPrePAMinBiasBHCOR + process.HLTEndSequence )
process.HLT_PAMinBiasHfOrBHC_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sPAMinBiasHfBptxANDorBscBptxAND + process.hltPrePAMinBiasHfOrBHC + process.HLTEndSequence )
process.HLT_PABptxPlusNotBptxMinus_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BptxPlusNotBptxMinus + process.hltPrePABptxPlusNotBptxMinus + process.HLTEndSequence )
process.HLT_PABptxMinusNotBptxPlus_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BptxMinusNotBptxPlus + process.hltPrePABptxMinusNotBptxPlus + process.HLTEndSequence )
process.HLT_PAZeroBias_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPrePAZeroBias + process.HLTEndSequence )
process.HLT_PAZeroBiasPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPrePAZeroBiasPixelSingleTrack + process.HLTDoLocalPixelSequence + process.HLTPixelTrackingForPAMinBiasSequence + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + process.HLTEndSequence )
process.HLT_PAHFOR_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sPAMinBiasHFORBptxAND + process.hltPrePAHFORSingleTrack + process.HLTDoLocalPixelSequence + process.HLTPixelTrackingForPAMinBiasSequence + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + process.HLTEndSequence )
process.HLT_PAZeroBiasPixel_DoubleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPrePAZeroBiasPixelDoubleTrack + process.HLTDoLocalPixelSequence + process.HLTPixelTrackingForPAMinBiasSequence + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter2 + process.HLTEndSequence )
process.HLT_PADoubleMu4_Acoplanarity03_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleMu0 + process.hltPrePADoubleMu4Acoplanarity03 + process.hltL1fL1sL1DoubleMu0L1f0 + process.HLTL2muonrecoSequence + process.hltL2fL1sL1DoubleMu0L1f0L2f0 + process.HLTL3muonrecoSequence + process.hltL3fL1sL1DoubleMu0L1f0L2f0L3f4 + process.hltDoubleMu4ExclL3PreFiltered + process.HLTEndSequence )
process.HLT_PAExclDijet35_HFOR_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet16BptxAND + process.hltPrePAExclDijet35HFOR + process.HLTRecoJetSequenceAK5Corrected + process.hltExclDiJet35HFOR + process.HLTEndSequence )
process.HLT_PAExclDijet35_HFAND_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleJet16FwdVeto5 + process.hltPrePAExclDijet35HFAND + process.HLTRecoJetSequenceAK5Corrected + process.hltExclDiJet35HFAND + process.HLTEndSequence )
process.HLT_PAL1DoubleEG3_FwdVeto_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleEG3FwdVeto + process.hltPrePAL1DoubleEG3FwdVeto + process.HLTEndSequence )
process.HLT_PAL1SingleJet52_TotemDiffractive_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet52TotemDiffractive + process.hltPrePAL1SingleJet52TotemDiffractive + process.HLTEndSequence )
process.HLT_PAL1SingleMu20_TotemDiffractive_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu20TotemDiffractive + process.hltPrePAL1SingleMu20TotemDiffractive + process.HLTEndSequence )
process.HLT_PAL1SingleEG20_TotemDiffractive_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG20TotemDiffractive + process.hltPrePAL1SingleEG20TotemDiffractive + process.HLTEndSequence )
process.HLT_PAL1DoubleJet20_TotemDiffractive_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleJet20TotemDiffractive + process.hltPrePAL1DoubleJet20TotemDiffractive + process.HLTEndSequence )
process.HLT_PAL1DoubleJetC36_TotemDiffractive_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleJetC36TotemDiffractive + process.hltPrePAL1DoubleJetC36TotemDiffractive + process.HLTEndSequence )
process.HLT_PAL1DoubleMu5_TotemDiffractive_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu5TotemDiffractive + process.hltPrePAL1DoubleMu5TotemDiffractive + process.HLTEndSequence )
process.HLT_PAL1DoubleEG5_TotemDiffractive_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5TotemDiffractive + process.hltPrePAL1DoubleEG5TotemDiffractive + process.HLTEndSequence )
process.HLT_PADoubleJet20_ForwardBackward_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1DoubleForJet16EtaOpp + process.hltPrePADoubleJet20ForwardBackward + process.HLTRecoJetSequenceAK5Corrected + process.hltDoubleJet20ForwardBackward + process.HLTEndSequence )
process.HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Mu0EG5 + process.hltPrePAMu7Ele7CaloIdTCaloIsoVL + process.hltL1Mu0EG5L1MuFiltered0 + process.HLTL2muonrecoSequence + process.hltL1Mu0EG5L2MuFiltered0 + process.HLTL3muonrecoSequence + process.hltL1Mu0EG5L3MuFiltered7 + process.HLTDoEGammaStartupSequence + process.hltEGRegionalL1Mu0EG5 + process.hltEG7EtFilterL1Mu0EG5 + process.HLTDoEgammaClusterShapeSequence + process.hltMu7Ele7CaloIdTCaloIsoVLClusterShapeFilter + process.HLTDoEGammaHESequence + process.hltMu7Ele7CaloIdTCaloIsoVLTHEFilter + process.hltL1SeededPhotonEcalIso + process.hltMu7Ele7CaloIdTCaloIsoVLEcalIsoFilter + process.hltL1SeededPhotonHcalIso + process.hltMu7Ele7CaloIdTCaloIsoVLHcalIsoFilter + process.HLTDoEGammaPixelSequence + process.hltMu7Ele7CaloIdTPixelMatchFilter + process.HLTEndSequence )
process.HLT_PAUpcSingleEG5Pixel_TrackVeto_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sPASingleEG5BptxAND + process.hltPrePAUpcSingleEG5PixelTrackVeto + process.HLTDoLocalPixelSequence + process.hltPAPixelTracksForMinBias + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + ~process.hltPACountPAPixFilter10 + process.HLTEndSequence )
process.HLT_PAUpcSingleEG5Full_TrackVeto7_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sPASingleEG5BptxAND + process.hltPrePAUpcSingleEG5FullTrackVeto7 + process.HLTDoLocalPixelSequence + process.hltPAPixelTracksForMinBias + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + ~process.hltPACountPAPixFilter10 + process.HLTPAUpcFullTrackRecoSequence + process.hltPACountUpcTrackFilter1 + ~process.hltPACountUpcTrackFilter7 + process.HLTEndSequence )
process.HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMuOpen + process.hltPrePAUpcSingleMuOpenPixelTrackVeto + process.HLTDoLocalPixelSequence + process.hltPAPixelTracksForMinBias + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + ~process.hltPACountPAPixFilter10 + process.HLTEndSequence )
process.HLT_PAUpcSingleMuOpenFull_TrackVeto7_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMuOpen + process.hltPrePAUpcSingleMuOpenFullTrackVeto7 + process.HLTDoLocalPixelSequence + process.hltPAPixelTracksForMinBias + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + ~process.hltPACountPAPixFilter10 + process.HLTPAUpcFullTrackRecoSequence + process.hltPACountUpcTrackFilter1 + ~process.hltPACountUpcTrackFilter7 + process.HLTEndSequence )
process.HLT_PAUpcSingleMuOpenTkMu_Onia_v1 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMuOpen + process.hltPrePAUpcSingleMuOpenTkMuOnia + process.HLTDoLocalPixelSequence + process.hltPAPixelTracksForMinBias + process.hltPAPixelCandsForMinBias + process.hltPAMinBiasPixelFilter1 + ~process.hltPACountPAPixFilter10 + process.hltPAUpcSingleMuOpenTkMuOniaDCAL1Filtered0 + process.HLTL2muonrecoSequence + process.hltPAUpcSingleMuOpenTkMuOniaDCAL2Filtered1 + process.HLTL3muonrecoSequence + process.hltPAUpcSingleMuOpenTkMuOniaDCAL3Filtered1 + process.HLTMuTrackUpcOniaPixelRecoSequence + process.hltMuOpenTrack1PixelMassFilteredUpcOnia + process.HLTMuTrackUpcOniaTrackRecoSequence + process.hltMuOpenTkMu1TrackMassFilteredUpcOnia + process.HLTMuTkMuUpcOniaTkMuRecoSequence + process.hltMuOpenTkMu1TkMuMassFilteredUpcOnia + process.HLTEndSequence )
process.HLT_PARandom_v1 = cms.Path( process.HLTBeginSequenceRandom + process.hltPrePARandom + process.HLTEndSequence )
process.DQM_FEDIntegrity_v11 = cms.Path( process.HLTBeginSequence + process.hltPreDQMFEDIntegrity + process.hltCSCMonitorModule + process.hltDTDQMEvF + process.hltEcalRawToRecHitFacility + process.hltEcalRegionalRestFEDs + process.hltEcalRecHitAll + process.hltEcalRawToRecHitByproductProducer + process.hltEBHltTask + process.hltEEHltTask + process.hltESFEDIntegrityTask + process.hltHcalDigis + process.hltHcalDataIntegrityMonitor + process.hltL1tfed + process.hltSiPixelDigis + process.hltSiPixelHLTSource + process.hltSiStripFEDCheck + process.hltMuonRPCDigis + process.hltRPCFEDIntegrity + process.hltBoolFalse )
process.HLT_LogMonitor_v4 = cms.Path( process.hltGtDigis + process.hltLogMonitorFilter + process.hltPreLogMonitor + process.HLTEndSequence )
process.HLTriggerFinalPath = cms.Path( process.hltGtDigis + process.hltScalersRawToDigi + process.hltFEDSelector + process.hltTriggerSummaryAOD + process.hltTriggerSummaryRAW )
process.HLTAnalyzerEndpath = cms.EndPath( process.hltL1GtTrigReport + process.hltTrigReport )
process.AOutput = cms.EndPath( process.hltPreAOutput + process.hltOutputA )
process.ALCAP0Output = cms.EndPath( process.hltPreALCAP0Output + process.hltOutputALCAP0 )
process.ALCAPHISYMOutput = cms.EndPath( process.hltPreALCAPHISYMOutput + process.hltOutputALCAPHISYM )
process.ALCALUMIPIXELSOutput = cms.EndPath( process.hltPreALCALUMIPIXELSOutput + process.hltOutputALCALUMIPIXELS )
process.CalibrationOutput = cms.EndPath( process.hltPreCalibrationOutput + process.hltOutputCalibration )
process.DQMOutput = cms.EndPath( process.hltPreDQMOutput + process.hltOutputDQM )
process.EcalCalibrationOutput = cms.EndPath( process.hltPreEcalCalibrationOutput + process.hltOutputEcalCalibration )
process.ExpressOutput = cms.EndPath( process.hltPreExpressOutput + process.hltPreExpressOutputSmart + process.hltOutputExpress )
process.HLTDQMOutput = cms.EndPath( process.hltPreHLTDQMOutput + process.hltPreHLTDQMOutputSmart + process.hltOutputHLTDQM )
process.NanoDSTOutput = cms.EndPath( process.hltPreNanoDSTOutput + process.hltOutputNanoDST )
process.RPCMONOutput = cms.EndPath( process.hltPreRPCMONOutput + process.hltOutputRPCMON )
process.TrackerCalibrationOutput = cms.EndPath( process.hltPreTrackerCalibrationOutput + process.hltOutputTrackerCalibration )


process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:RelVal_Raw_PIon_STARTUP.root',
    ),
    secondaryFileNames = cms.untracked.vstring(
    ),
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

# customise the HLT menu for running on MC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
process = customizeHLTforMC(process)

# remove the HLT prescales
if 'PrescaleService' in process.__dict__:
    process.PrescaleService.lvl1DefaultLabel = cms.string( '0' )
    process.PrescaleService.lvl1Labels       = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    process.PrescaleService.prescaleTable    = cms.VPSet( )

# CMSSW version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

# customization for CMSSW_5_2_X
if cmsswVersion.startswith('CMSSW_5_2_'):

    # force the use of the correct calo jet energy corrections
    if 'hltESPL1FastJetCorrectionESProducer' in process.__dict__:
        process.hltESPL1FastJetCorrectionESProducer.algorithm  = "AK5CaloHLT"

    if 'hltESPL2RelativeCorrectionESProducer' in process.__dict__:
        process.hltESPL2RelativeCorrectionESProducer.algorithm = "AK5CaloHLT"

    if 'hltESPL3AbsoluteCorrectionESProducer' in process.__dict__:
        process.hltESPL3AbsoluteCorrectionESProducer.algorithm = "AK5CaloHLT"


# customization for CMSSW_5_3_X
if cmsswVersion.startswith('CMSSW_5_3_'):

    # do not override the calo jet energy corrections in 5.3.x for consistency with the current MC samples
    pass


# customization for CMSSW_6_1_X and 6_2_X
if cmsswVersion.startswith('CMSSW_6_1_') or cmsswVersion.startswith('CMSSW_6_2_'):

    # force the use of the correct calo jet energy corrections
    if 'hltESPL1FastJetCorrectionESProducer' in process.__dict__:
        process.hltESPL1FastJetCorrectionESProducer.algorithm  = "AK5CaloHLT"

    if 'hltESPL2RelativeCorrectionESProducer' in process.__dict__:
        process.hltESPL2RelativeCorrectionESProducer.algorithm = "AK5CaloHLT"

    if 'hltESPL3AbsoluteCorrectionESProducer' in process.__dict__:
        process.hltESPL3AbsoluteCorrectionESProducer.algorithm = "AK5CaloHLT"

    # adapt the HLT menu to the "prototype for Event Interpretation" development
    if 'hltPFPileUp' in process.__dict__:
        # define new PFCandidateFwdPtrProducer module
        process.hltParticleFlowPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
            src = cms.InputTag('hltParticleFlow')
        )
        # add the new module before the hltPFPileUp module
        _sequence = None
        for _sequence in [ _sequence for _sequence in process.__dict__.itervalues() if isinstance(_sequence, cms._ModuleSequenceType)]:
            try:
                _sequence.insert( _sequence.index(process.hltPFPileUp), process.hltParticleFlowPtrs )
            except ValueError:
                pass
        # reconfigure hltPFPileUp and hltPFNoPileUp to use the new module
        process.hltPFPileUp.PFCandidates       = cms.InputTag( "hltParticleFlowPtrs" )
        process.hltPFNoPileUp.bottomCollection = cms.InputTag( "hltParticleFlowPtrs" )

    # postLS1 muon extension
    # /CalibMuon/CSCCalibration/python/CSCIndexer_cfi.py
    process.CSCIndexerESSource = cms.ESSource("EmptyESSource",
      recordName = cms.string("CSCIndexerRecord"),
      firstValid = cms.vuint32(1),
      iovIsRunNotTime = cms.bool(True)
    )
    process.CSCIndexerESProducer = cms.ESProducer("CSCIndexerESProducer",
      AlgoName = cms.string("CSCIndexerStartup")
    )
    # /CalibMuon/CSCCalibration/python/CSCChannelMapper_cfi.py
    process.CSCChannelMapperESSource = cms.ESSource("EmptyESSource",
      recordName = cms.string("CSCChannelMapperRecord"),
      firstValid = cms.vuint32(1),
      iovIsRunNotTime = cms.bool(True)
    )
    process.CSCChannelMapperESProducer = cms.ESProducer("CSCChannelMapperESProducer",
      AlgoName = cms.string("CSCChannelMapperStartup")
    )

# customization for CMSSW_6_2_X only
if cmsswVersion.startswith('CMSSW_6_2_'):
    # /Geometry/TrackerNumberingBuilder/trackerTopologyConstants_cfi.py
    process.trackerTopologyConstants = cms.ESProducer('TrackerTopologyEP',
      pxb_layerStartBit = cms.uint32(16),
      pxb_ladderStartBit = cms.uint32(8),
      pxb_moduleStartBit = cms.uint32(2),
      pxb_layerMask = cms.uint32(15),
      pxb_ladderMask = cms.uint32(255),
      pxb_moduleMask = cms.uint32(63),
      pxf_sideStartBit = cms.uint32(23),
      pxf_diskStartBit = cms.uint32(16),
      pxf_bladeStartBit = cms.uint32(10),
      pxf_panelStartBit = cms.uint32(8),
      pxf_moduleStartBit = cms.uint32(2),
      pxf_sideMask = cms.uint32(3),
      pxf_diskMask = cms.uint32(15),
      pxf_bladeMask = cms.uint32(63),
      pxf_panelMask = cms.uint32(3),
      pxf_moduleMask = cms.uint32(63),
      tec_sideStartBit = cms.uint32(18),
      tec_wheelStartBit = cms.uint32(14),
      tec_petal_fw_bwStartBit = cms.uint32(12),
      tec_petalStartBit = cms.uint32(8),
      tec_ringStartBit = cms.uint32(5),
      tec_moduleStartBit = cms.uint32(2),
      tec_sterStartBit = cms.uint32(0),
      tec_sideMask = cms.uint32(3),
      tec_wheelMask = cms.uint32(15),
      tec_petal_fw_bwMask = cms.uint32(3),
      tec_petalMask = cms.uint32(15),
      tec_ringMask = cms.uint32(7),
      tec_moduleMask = cms.uint32(7),
      tec_sterMask = cms.uint32(3),
      tib_layerStartBit = cms.uint32(14),
      tib_str_fw_bwStartBit = cms.uint32(12),
      tib_str_int_extStartBit = cms.uint32(10),
      tib_strStartBit = cms.uint32(4),
      tib_moduleStartBit = cms.uint32(2),
      tib_sterStartBit = cms.uint32(0),
      tib_layerMask = cms.uint32(7),
      tib_str_fw_bwMask = cms.uint32(3),
      tib_str_int_extMask = cms.uint32(3),
      tib_strMask = cms.uint32(63),
      tib_moduleMask = cms.uint32(3),
      tib_sterMask = cms.uint32(3),
      tid_sideStartBit = cms.uint32(13),
      tid_wheelStartBit = cms.uint32(11),
      tid_ringStartBit = cms.uint32(9),
      tid_module_fw_bwStartBit = cms.uint32(7),
      tid_moduleStartBit = cms.uint32(2),
      tid_sterStartBit = cms.uint32(0),
      tid_sideMask = cms.uint32(3),
      tid_wheelMask = cms.uint32(3),
      tid_ringMask = cms.uint32(3),
      tid_module_fw_bwMask = cms.uint32(3),
      tid_moduleMask = cms.uint32(31),
      tid_sterMask = cms.uint32(3),
      tob_layerStartBit = cms.uint32(14),
      tob_rod_fw_bwStartBit = cms.uint32(12),
      tob_rodStartBit = cms.uint32(5),
      tob_moduleStartBit = cms.uint32(2),
      tob_sterStartBit = cms.uint32(0),
      tob_layerMask = cms.uint32(7),
      tob_rod_fw_bwMask = cms.uint32(3),
      tob_rodMask = cms.uint32(127),
      tob_moduleMask = cms.uint32(7),
      tob_sterMask = cms.uint32(3),
      appendToDataLabel = cms.string('')
    )


# adapt HLT modules to the correct process name
if 'hltTrigReport' in process.__dict__:
    process.hltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltPreExpressCosmicsOutputSmart' in process.__dict__:
    process.hltPreExpressCosmicsOutputSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltPreExpressOutputSmart' in process.__dict__:
    process.hltPreExpressOutputSmart.TriggerResultsTag        = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltPreDQMForHIOutputSmart' in process.__dict__:
    process.hltPreDQMForHIOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltPreDQMForPPOutputSmart' in process.__dict__:
    process.hltPreDQMForPPOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltPreHLTDQMResultsOutputSmart' in process.__dict__:
    process.hltPreHLTDQMResultsOutputSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltPreHLTDQMOutputSmart' in process.__dict__:
    process.hltPreHLTDQMOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltPreHLTMONOutputSmart' in process.__dict__:
    process.hltPreHLTMONOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )

if 'hltDQMHLTScalers' in process.__dict__:
    process.hltDQMHLTScalers.triggerResults                   = cms.InputTag( 'TriggerResults', '', 'HLTPIon' )
    process.hltDQMHLTScalers.processname                      = 'HLTPIon'

if 'hltDQML1SeedLogicScalers' in process.__dict__:
    process.hltDQML1SeedLogicScalers.processname              = 'HLTPIon'

# limit the number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

# enable the TrigReport and TimeReport
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
    process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = 'auto:startup_PIon', conditions = 'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T')
    process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
    for pset in process.GlobalTag.toGet.value():
        pset.connect = pset.connect.value().replace('frontier://FrontierProd/', 'frontier://FrontierProd/')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')
    process.MessageLogger.categories.append('FastReport')

