# /dev/CMSSW_5_2_0/HIon/V17

import FWCore.ParameterSet.Config as cms


# dump of the Stream AForHI Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHIDiMuon_selector
streamA_datasetHIDiMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHIDiMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetHIDiMuon_selector.throw      = cms.bool(False)
streamA_datasetHIDiMuon_selector.triggerConditions = cms.vstring('HLT_HIL1DoubleMu0_HighQ_v3', 
    'HLT_HIL1DoubleMuOpen_v3', 
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v3', 
    'HLT_HIL2DoubleMu0_NHitQ_v3', 
    'HLT_HIL2DoubleMu0_v3', 
    'HLT_HIL2DoubleMu3_v3', 
    'HLT_HIL2Mu15_v3', 
    'HLT_HIL2Mu3_NHitQ_v3', 
    'HLT_HIL2Mu3_v3', 
    'HLT_HIL2Mu7_v3', 
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v3', 
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v3', 
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v3', 
    'HLT_HIL3DoubleMuOpen_Mgt2_v3', 
    'HLT_HIL3DoubleMuOpen_v3', 
    'HLT_HIL3Mu3_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHIHighPt_selector
streamA_datasetHIHighPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHIHighPt_selector.l1tResults = cms.InputTag('')
streamA_datasetHIHighPt_selector.throw      = cms.bool(False)
streamA_datasetHIHighPt_selector.triggerConditions = cms.vstring('HLT_HIDiJet55_v2', 
    'HLT_HIDoublePhoton10_v2', 
    'HLT_HIDoublePhoton15_v2', 
    'HLT_HIDoublePhoton20_v2', 
    'HLT_HIFullTrack12_L1Central_v2', 
    'HLT_HIFullTrack12_L1Peripheral_v2', 
    'HLT_HIFullTrack14_L1Central_v2', 
    'HLT_HIFullTrack14_L1Peripheral_v2', 
    'HLT_HIFullTrack20_L1Central_v2', 
    'HLT_HIFullTrack20_L1Peripheral_v2', 
    'HLT_HIFullTrack25_L1Central_v2', 
    'HLT_HIFullTrack25_L1Peripheral_v2', 
    'HLT_HIJet55_v2', 
    'HLT_HIJet65_Jet55_v2', 
    'HLT_HIJet65_v2', 
    'HLT_HIJet80_v2', 
    'HLT_HIJet95_v2', 
    'HLT_HIJetE30_NoBPTX_v2', 
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v2', 
    'HLT_HIMET120_v2', 
    'HLT_HIMET200_v2', 
    'HLT_HIMET220_v2', 
    'HLT_HIPhoton10_Photon15_v2', 
    'HLT_HIPhoton15_Photon20_v2', 
    'HLT_HISinglePhoton15_v2', 
    'HLT_HISinglePhoton20_v3', 
    'HLT_HISinglePhoton30_v3', 
    'HLT_HISinglePhoton40_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHIMinBiasUPC_selector
streamA_datasetHIMinBiasUPC_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHIMinBiasUPC_selector.l1tResults = cms.InputTag('')
streamA_datasetHIMinBiasUPC_selector.throw      = cms.bool(False)
streamA_datasetHIMinBiasUPC_selector.triggerConditions = cms.vstring('HLT_HIBptxXOR_v2', 
    'HLT_HICentral10_v3', 
    'HLT_HICentralityVeto_v2', 
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v2', 
    'HLT_HIMinBiasBSC_OR_v2', 
    'HLT_HIMinBiasBSC_v2', 
    'HLT_HIMinBiasHF_v2', 
    'HLT_HIMinBiasHfOrBSC_v2', 
    'HLT_HIMinBiasHf_OR_v2', 
    'HLT_HIMinBiasPixel_SingleTrack_v2', 
    'HLT_HIMinBiasZDCPixel_SingleTrack_v2', 
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v2', 
    'HLT_HIMinBiasZDC_Calo_v2', 
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v2', 
    'HLT_HIPhysics_v2', 
    'HLT_HIRandom_v2', 
    'HLT_HIUCC010_v3', 
    'HLT_HIUCC015_v3', 
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v2', 
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v2', 
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v2', 
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v2', 
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v2', 
    'HLT_HIUPCNeuMuPixel_SingleTrack_v2', 
    'HLT_HIZeroBiasPixel_SingleTrack_v2', 
    'HLT_HIZeroBiasXOR_v2', 
    'HLT_HIZeroBias_v2')

