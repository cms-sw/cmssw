# /dev/CMSSW_4_4_2/HIon/V41

import FWCore.ParameterSet.Config as cms

# dump of the Stream A Datasets defined in the HLT table

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHIDiMuon_selector
streamA_datasetHIDiMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHIDiMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetHIDiMuon_selector.throw      = cms.bool(False)
streamA_datasetHIDiMuon_selector.triggerConditions = cms.vstring('HLT_HIL1DoubleMu0_HighQ_v2', 
    'HLT_HIL1DoubleMuOpen_v2', 
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2', 
    'HLT_HIL2DoubleMu0_NHitQ_v2', 
    'HLT_HIL2DoubleMu0_v2', 
    'HLT_HIL2DoubleMu3_v2', 
    'HLT_HIL2Mu15_v2', 
    'HLT_HIL2Mu3_NHitQ_v2', 
    'HLT_HIL2Mu3_v2', 
    'HLT_HIL2Mu7_v2', 
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2', 
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v2', 
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v2', 
    'HLT_HIL3DoubleMuOpen_Mgt2_v2', 
    'HLT_HIL3DoubleMuOpen_v2', 
    'HLT_HIL3Mu3_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHIHighPt_selector
streamA_datasetHIHighPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHIHighPt_selector.l1tResults = cms.InputTag('')
streamA_datasetHIHighPt_selector.throw      = cms.bool(False)
streamA_datasetHIHighPt_selector.triggerConditions = cms.vstring('HLT_HIDiJet55_v1', 
    'HLT_HIDoublePhoton10_v1', 
    'HLT_HIDoublePhoton15_v1', 
    'HLT_HIDoublePhoton20_v1', 
    'HLT_HIFullTrack12_L1Central_v1', 
    'HLT_HIFullTrack12_L1Peripheral_v1', 
    'HLT_HIFullTrack14_L1Central_v1', 
    'HLT_HIFullTrack14_L1Peripheral_v1', 
    'HLT_HIFullTrack20_L1Central_v1', 
    'HLT_HIFullTrack20_L1Peripheral_v1', 
    'HLT_HIFullTrack25_L1Central_v1', 
    'HLT_HIFullTrack25_L1Peripheral_v1', 
    'HLT_HIJet55_v1', 
    'HLT_HIJet65_Jet55_v1', 
    'HLT_HIJet65_v1', 
    'HLT_HIJet80_v1', 
    'HLT_HIJet95_v1', 
    'HLT_HIJetE30_NoBPTX_v1', 
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v1', 
    'HLT_HIMET120_v1', 
    'HLT_HIMET200_v1', 
    'HLT_HIMET220_v1', 
    'HLT_HIPhoton10_Photon15_v1', 
    'HLT_HIPhoton15_Photon20_v1', 
    'HLT_HISinglePhoton15_v1', 
    'HLT_HISinglePhoton20_v2', 
    'HLT_HISinglePhoton30_v2', 
    'HLT_HISinglePhoton40_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHIMinBiasUPC_selector
streamA_datasetHIMinBiasUPC_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHIMinBiasUPC_selector.l1tResults = cms.InputTag('')
streamA_datasetHIMinBiasUPC_selector.throw      = cms.bool(False)
streamA_datasetHIMinBiasUPC_selector.triggerConditions = cms.vstring('HLT_HIBptxXOR_v1', 
    'HLT_HICentral10_v2', 
    'HLT_HICentralityVeto_v1', 
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v1', 
    'HLT_HIMinBiasBSC_OR_v1', 
    'HLT_HIMinBiasBSC_v1', 
    'HLT_HIMinBiasHF_v1', 
    'HLT_HIMinBiasHfOrBSC_v1', 
    'HLT_HIMinBiasHf_OR_v1', 
    'HLT_HIMinBiasPixel_SingleTrack_v1', 
    'HLT_HIMinBiasZDCPixel_SingleTrack_v1', 
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1', 
    'HLT_HIMinBiasZDC_Calo_v1', 
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1', 
    'HLT_HIPhysics_v1', 
    'HLT_HIRandom_v1', 
    'HLT_HIUCC010_v2', 
    'HLT_HIUCC015_v2', 
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v1', 
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v1', 
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1', 
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1', 
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1', 
    'HLT_HIUPCNeuMuPixel_SingleTrack_v1', 
    'HLT_HIZeroBiasPixel_SingleTrack_v1', 
    'HLT_HIZeroBiasXOR_v1', 
    'HLT_HIZeroBias_v1')

