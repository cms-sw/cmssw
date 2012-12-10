# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v13', 
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
    'HLT_PAHcalPhiSym_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetMon_selector
streamA_datasetJetMon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetMon_selector.l1tResults = cms.InputTag('')
streamA_datasetJetMon_selector.throw      = cms.bool(False)
streamA_datasetJetMon_selector.triggerConditions = cms.vstring('HLT_PASingleForJet15_v1', 
    'HLT_PASingleForJet25_v1')

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
streamA_datasetPAHighPt_selector.triggerConditions = cms.vstring('HLT_PAForJet40Eta2_v1', 
    'HLT_PAForJet40Eta3_v1', 
    'HLT_PAForJet60Eta2_v1', 
    'HLT_PAForJet60Eta3_v1', 
    'HLT_PAForJet80Eta2_v1', 
    'HLT_PAForJet80Eta3_v1', 
    'HLT_PAFullTrack12_v1', 
    'HLT_PAFullTrack20_v1', 
    'HLT_PAFullTrack30_v1', 
    'HLT_PAFullTrack50_v1', 
    'HLT_PAJet100_NoJetID_v1', 
    'HLT_PAJet40ETM30_v1', 
    'HLT_PAJet40_NoJetID_v1', 
    'HLT_PAJet60_NoJetID_v1', 
    'HLT_PAJet80_NoJetID_v1', 
    'HLT_PAPhoton10_NoCaloIdVL_v1', 
    'HLT_PAPhoton10_Photon10_NoCaloIdVL_v1', 
    'HLT_PAPhoton15_Photon10_NoCaloIdVL_v1', 
    'HLT_PAPhoton20_NoCaloIdVL_v1', 
    'HLT_PAPhoton20_Photon15_NoCaloIdVL_v1', 
    'HLT_PAPhoton20_Photon20_NoCaloIdVL_v1', 
    'HLT_PAPhoton30_NoCaloIdVL_v1', 
    'HLT_PAPhoton30_Photon30_NoCaloIdVL_v1', 
    'HLT_PAPhoton40_NoCaloIdVL_v1', 
    'HLT_PAPhoton60_NoCaloIdVL_v1', 
    'HLT_PAPixelTrackMultiplicity100_FullTrack12_v1', 
    'HLT_PAPixelTrackMultiplicity130_FullTrack12_v1', 
    'HLT_PAPixelTrackMultiplicity160_FullTrack12_v1', 
    'HLT_PAPixelTracks_Multiplicity100_v1', 
    'HLT_PAPixelTracks_Multiplicity130_v1', 
    'HLT_PAPixelTracks_Multiplicity160_v1', 
    'HLT_PAPixelTracks_Multiplicity190_v1', 
    'HLT_PAPixelTracks_Multiplicity220_v1', 
    'HLT_PATripleJet40_20_20_v1', 
    'HLT_PATripleJet60_20_20_v1', 
    'HLT_PATripleJet80_20_20_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAMinBiasUPC_selector
streamA_datasetPAMinBiasUPC_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAMinBiasUPC_selector.l1tResults = cms.InputTag('')
streamA_datasetPAMinBiasUPC_selector.throw      = cms.bool(False)
streamA_datasetPAMinBiasUPC_selector.triggerConditions = cms.vstring('HLT_PABptxMinusNotBptxPlus_v1', 
    'HLT_PABptxPlusNotBptxMinus_v1', 
    'HLT_PADimuon0_NoVertexing_v1', 
    'HLT_PADoubleJet20_ForwardBackward_v1', 
    'HLT_PADoubleMu4_Acoplanarity03_v1', 
    'HLT_PAExclDijet35_HFAND_v1', 
    'HLT_PAHFOR_SingleTrack_v1', 
    'HLT_PAL1DoubleEG3_FwdVeto_v1', 
    'HLT_PAL1DoubleJet20_TotemDiffractive_v1', 
    'HLT_PAL1DoubleMu0_v1', 
    'HLT_PAL1SingleJet16_v1', 
    'HLT_PAL1SingleJet36_v1', 
    'HLT_PAL1Tech53_MB_SingleTrack_v1', 
    'HLT_PAL1Tech53_MB_v1', 
    'HLT_PAL1Tech54_ZeroBias_v1', 
    'HLT_PAL1Tech_HBHEHO_totalOR_v1', 
    'HLT_PAMinBiasBSC_OR_v1', 
    'HLT_PAMinBiasBSC_v1', 
    'HLT_PAMinBiasHF_OR_v1', 
    'HLT_PAMinBiasHF_v1', 
    'HLT_PAMinBiasHfOrBSC_v1', 
    'HLT_PAMu7_Ele7_CaloIdT_CaloIsoVL_v1', 
    'HLT_PARandom_v1', 
    'HLT_PARomanPots_Tech52_v1', 
    'HLT_PASingleForJet15_v1', 
    'HLT_PASingleForJet25_v1', 
    'HLT_PAT1minbias_Tech55_v1', 
    'HLT_PAUpcSingleEG5Pixel_TrackVeto_v1', 
    'HLT_PAUpcSingleMuOpenPixel_TrackVeto_v1', 
    'HLT_PAUpcSingleMuOpenTkMu_Onia_v1', 
    'HLT_PAZeroBiasPixel_DoubleTrack_v1', 
    'HLT_PAZeroBiasPixel_SingleTrack_v1', 
    'HLT_PAZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAMuon_selector
streamA_datasetPAMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetPAMuon_selector.throw      = cms.bool(False)
streamA_datasetPAMuon_selector.triggerConditions = cms.vstring('HLT_PABTagMu_Jet20_Mu4_v1', 
    'HLT_PAL1DoubleMu0_HighQ_v1', 
    'HLT_PAL1DoubleMuOpen_v1', 
    'HLT_PAL2DoubleMu3_v1', 
    'HLT_PAMu12_v1', 
    'HLT_PAMu3PFJet20_v1', 
    'HLT_PAMu3PFJet40_v1', 
    'HLT_PAMu3_v1', 
    'HLT_PAMu7PFJet20_v1', 
    'HLT_PAMu7_v1')

