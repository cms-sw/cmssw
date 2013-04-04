# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v14', 
    'HLT_BeamGas_HF_Beam1_v5', 
    'HLT_BeamGas_HF_Beam2_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCosmics_selector
streamA_datasetCosmics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCosmics_selector.l1tResults = cms.InputTag('')
streamA_datasetCosmics_selector.throw      = cms.bool(False)
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v13', 
    'HLT_L1SingleMuOpen_AntiBPTX_v7', 
    'HLT_L1TrackerCosmics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_PAHcalNZS_v1', 
    'HLT_PAHcalPhiSym_v1', 
    'HLT_PAHcalUTCA_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetLogMonitor_selector
streamA_datasetLogMonitor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetLogMonitor_selector.l1tResults = cms.InputTag('')
streamA_datasetLogMonitor_selector.throw      = cms.bool(False)
streamA_datasetLogMonitor_selector.triggerConditions = cms.vstring('HLT_LogMonitor_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_Physics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAHighPt_selector
streamA_datasetPAHighPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAHighPt_selector.l1tResults = cms.InputTag('')
streamA_datasetPAHighPt_selector.throw      = cms.bool(False)
streamA_datasetPAHighPt_selector.triggerConditions = cms.vstring('HLT_PADoubleEle6_CaloIdT_TrkIdVL_v2', 
    'HLT_PADoubleEle8_CaloIdT_TrkIdVL_v2', 
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
    'HLT_PAFullTrack12_v3', 
    'HLT_PAFullTrack20_v3', 
    'HLT_PAFullTrack30_v3', 
    'HLT_PAFullTrack50_v3', 
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
    'HLT_PAL1DoubleEG5DoubleEle6_CaloIdT_TrkIdVL_v2', 
    'HLT_PAPhoton10_NoCaloIdVL_v2', 
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v2', 
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_v2', 
    'HLT_PAPhoton10_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton10_TightCaloIdVL_v2', 
    'HLT_PAPhoton15_NoCaloIdVL_v2', 
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v2', 
    'HLT_PAPhoton15_Photon10_TightCaloIdVL_v2', 
    'HLT_PAPhoton15_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton15_TightCaloIdVL_v2', 
    'HLT_PAPhoton20_NoCaloIdVL_v2', 
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v2', 
    'HLT_PAPhoton20_Photon15_TightCaloIdVL_v2', 
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v2', 
    'HLT_PAPhoton20_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton20_TightCaloIdVL_v2', 
    'HLT_PAPhoton30_NoCaloIdVL_v2', 
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v2', 
    'HLT_PAPhoton30_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton30_TightCaloIdVL_v2', 
    'HLT_PAPhoton40_NoCaloIdVL_v2', 
    'HLT_PAPhoton40_TightCaloIdVL_v2', 
    'HLT_PAPhoton60_NoCaloIdVL_v2', 
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v3', 
    'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v2', 
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v3', 
    'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v3', 
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v3', 
    'HLT_PAPixelTracks_Multiplicity100_v3', 
    'HLT_PAPixelTracks_Multiplicity130_v3', 
    'HLT_PAPixelTracks_Multiplicity160_v3', 
    'HLT_PAPixelTracks_Multiplicity190_v3', 
    'HLT_PAPixelTracks_Multiplicity220_v3', 
    'HLT_PASingleEle6_CaloIdNone_TrkIdVL_v2', 
    'HLT_PASingleEle6_CaloIdT_TrkIdVL_v2', 
    'HLT_PASingleEle8_CaloIdNone_TrkIdVL_v2', 
    'HLT_PATech35_HFSumET100_v3', 
    'HLT_PATech35_v1', 
    'HLT_PATripleJet100_20_20_v1', 
    'HLT_PATripleJet20_20_20_v1', 
    'HLT_PATripleJet40_20_20_v1', 
    'HLT_PATripleJet60_20_20_v1', 
    'HLT_PATripleJet80_20_20_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAMinBiasUPC_selector
streamA_datasetPAMinBiasUPC_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAMinBiasUPC_selector.l1tResults = cms.InputTag('')
streamA_datasetPAMinBiasUPC_selector.throw      = cms.bool(False)
streamA_datasetPAMinBiasUPC_selector.triggerConditions = cms.vstring('HLT_PABptxMinusNotBptxPlus_v1', 
    'HLT_PABptxPlusNotBptxMinus_v1', 
    'HLT_PACastorEmNotHfCoincidencePm_v1', 
    'HLT_PACastorEmNotHfSingleChannel_v1', 
    'HLT_PACastorEmTotemLowMultiplicity_v1', 
    'HLT_PADimuon0_NoVertexing_v1', 
    'HLT_PADoubleJet20_ForwardBackward_v1', 
    'HLT_PADoubleMu4_Acoplanarity03_v2', 
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
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v2', 
    'HLT_PARandom_v1', 
    'HLT_PARomanPots_Tech52_v1', 
    'HLT_PASingleForJet15_v1', 
    'HLT_PASingleForJet25_v1', 
    'HLT_PAT1minbias_Tech55_v1', 
    'HLT_PAUpcSingleEG5Full_TrackVeto7_v2', 
    'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1', 
    'HLT_PAUpcSingleMuOpenFull_TrackVeto7_v2', 
    'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1', 
    'HLT_PAUpcSingleMuOpenTkMu_Onia_v2', 
    'HLT_PAZeroBiasPixel_DoubleTrack_v1', 
    'HLT_PAZeroBiasPixel_SingleTrack_v1', 
    'HLT_PAZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAMuon_selector
streamA_datasetPAMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetPAMuon_selector.throw      = cms.bool(False)
streamA_datasetPAMuon_selector.triggerConditions = cms.vstring('HLT_PABTagMu_Jet20_Mu4_v2', 
    'HLT_PAL1DoubleMu0_HighQ_v1', 
    'HLT_PAL1DoubleMuOpen_v1', 
    'HLT_PAL2DoubleMu3_v1', 
    'HLT_PAMu12_v2', 
    'HLT_PAMu3PFJet20_v2', 
    'HLT_PAMu3PFJet40_v2', 
    'HLT_PAMu3_v2', 
    'HLT_PAMu7PFJet20_v2', 
    'HLT_PAMu7_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPPFSQ_selector
streamA_datasetPPFSQ_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPPFSQ_selector.l1tResults = cms.InputTag('')
streamA_datasetPPFSQ_selector.throw      = cms.bool(False)
streamA_datasetPPFSQ_selector.triggerConditions = cms.vstring('HLT_PADimuon0_NoVertexing_v1', 
    'HLT_PADoubleJet20_ForwardBackward_v1', 
    'HLT_PADoubleMu4_Acoplanarity03_v2', 
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
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v2', 
    'HLT_PASingleForJet15_v1', 
    'HLT_PASingleForJet25_v1', 
    'HLT_PPL1DoubleJetC36_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPPJet_selector
streamA_datasetPPJet_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPPJet_selector.l1tResults = cms.InputTag('')
streamA_datasetPPJet_selector.throw      = cms.bool(False)
streamA_datasetPPJet_selector.triggerConditions = cms.vstring('HLT_PAForJet100Eta2_v1', 
    'HLT_PAForJet100Eta3_v1', 
    'HLT_PAForJet20Eta2_v1', 
    'HLT_PAForJet20Eta3_v1', 
    'HLT_PAForJet40Eta2_v1', 
    'HLT_PAForJet40Eta3_v1', 
    'HLT_PAForJet60Eta2_v1', 
    'HLT_PAForJet60Eta3_v1', 
    'HLT_PAForJet80Eta2_v1', 
    'HLT_PAForJet80Eta3_v1', 
    'HLT_PAFullTrack12_v3', 
    'HLT_PAFullTrack20_v3', 
    'HLT_PAFullTrack30_v3', 
    'HLT_PAFullTrack50_v3', 
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
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v3', 
    'HLT_PAPixelTrackMultiplicity100_L2DoubleMu3_v2', 
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v3', 
    'HLT_PAPixelTrackMultiplicity140_Jet80_NoJetID_v3', 
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v3', 
    'HLT_PAPixelTracks_Multiplicity100_v3', 
    'HLT_PAPixelTracks_Multiplicity130_v3', 
    'HLT_PAPixelTracks_Multiplicity160_v3', 
    'HLT_PAPixelTracks_Multiplicity190_v3', 
    'HLT_PAPixelTracks_Multiplicity220_v3', 
    'HLT_PATech35_HFSumET100_v3', 
    'HLT_PATech35_v1', 
    'HLT_PATripleJet100_20_20_v1', 
    'HLT_PATripleJet20_20_20_v1', 
    'HLT_PATripleJet40_20_20_v1', 
    'HLT_PATripleJet60_20_20_v1', 
    'HLT_PATripleJet80_20_20_v1', 
    'HLT_PPPixelTrackMultiplicity55_FullTrack12_v2', 
    'HLT_PPPixelTrackMultiplicity70_FullTrack12_v2', 
    'HLT_PPPixelTracks_Multiplicity55_v2', 
    'HLT_PPPixelTracks_Multiplicity70_v2', 
    'HLT_PPPixelTracks_Multiplicity85_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPPMinBias_selector
streamA_datasetPPMinBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPPMinBias_selector.l1tResults = cms.InputTag('')
streamA_datasetPPMinBias_selector.throw      = cms.bool(False)
streamA_datasetPPMinBias_selector.triggerConditions = cms.vstring('HLT_PABptxMinusNotBptxPlus_v1', 
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
    'HLT_PAZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPPMuon_selector
streamA_datasetPPMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPPMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetPPMuon_selector.throw      = cms.bool(False)
streamA_datasetPPMuon_selector.triggerConditions = cms.vstring('HLT_Mu15_eta2p1_v6', 
    'HLT_PABTagMu_Jet20_Mu4_v2', 
    'HLT_PAL1DoubleMu0_HighQ_v1', 
    'HLT_PAL1DoubleMuOpen_v1', 
    'HLT_PAL2DoubleMu3_v1', 
    'HLT_PAMu12_v2', 
    'HLT_PAMu3PFJet20_v2', 
    'HLT_PAMu3PFJet40_v2', 
    'HLT_PAMu3_v2', 
    'HLT_PAMu7PFJet20_v2', 
    'HLT_PAMu7_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPPPhoton_selector
streamA_datasetPPPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPPPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetPPPhoton_selector.throw      = cms.bool(False)
streamA_datasetPPPhoton_selector.triggerConditions = cms.vstring('HLT_Ele22_CaloIdL_CaloIsoVL_v7', 
    'HLT_PAPhoton10_NoCaloIdVL_v2', 
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v2', 
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton10_Photon10_TightCaloIdVL_v2', 
    'HLT_PAPhoton10_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton10_TightCaloIdVL_v2', 
    'HLT_PAPhoton15_NoCaloIdVL_v2', 
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v2', 
    'HLT_PAPhoton15_Photon10_TightCaloIdVL_v2', 
    'HLT_PAPhoton15_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton15_TightCaloIdVL_v2', 
    'HLT_PAPhoton20_NoCaloIdVL_v2', 
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v2', 
    'HLT_PAPhoton20_Photon15_TightCaloIdVL_v2', 
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v2', 
    'HLT_PAPhoton20_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton20_TightCaloIdVL_v2', 
    'HLT_PAPhoton30_NoCaloIdVL_v2', 
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v2', 
    'HLT_PAPhoton30_TightCaloIdVL_Iso50_v2', 
    'HLT_PAPhoton30_TightCaloIdVL_v2', 
    'HLT_PAPhoton40_NoCaloIdVL_v2', 
    'HLT_PAPhoton40_TightCaloIdVL_v2', 
    'HLT_PAPhoton60_NoCaloIdVL_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleElectron_selector
streamA_datasetSingleElectron_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleElectron_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleElectron_selector.throw      = cms.bool(False)
streamA_datasetSingleElectron_selector.triggerConditions = cms.vstring('HLT_Ele22_CaloIdL_CaloIsoVL_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMu_selector
streamA_datasetSingleMu_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMu_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMu_selector.throw      = cms.bool(False)
streamA_datasetSingleMu_selector.triggerConditions = cms.vstring('HLT_Mu15_eta2p1_v6')

