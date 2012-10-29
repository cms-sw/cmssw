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
streamA_datasetCosmics_selector.triggerConditions = cms.vstring('HLT_BeamHalo_v13')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v10', 
    'HLT_HcalPhiSym_v11')

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

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAPhysics_selector
streamA_datasetPAPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAPhysics_selector.l1tResults = cms.InputTag('')
streamA_datasetPAPhysics_selector.throw      = cms.bool(False)
streamA_datasetPAPhysics_selector.triggerConditions = cms.vstring('HLT_PABTagMu_Jet20_Mu4_v1', 
    'HLT_PADimuon0_NoVertexing_v1', 
    'HLT_PADoubleJet20_ForwardBackward_v1', 
    'HLT_PADoubleMu4_Acoplanarity03_v1', 
    'HLT_PAExclDijet35_HFAND_v1', 
    'HLT_PAHFOR_SingleTrack_v1', 
    'HLT_PAL1DoubleEG3_FwdVeto_v1', 
    'HLT_PAL1DoubleJet20_RomanPotsOR_v1', 
    'HLT_PAL1DoubleMu0_HighQ_v1', 
    'HLT_PAL1DoubleMu0_v1', 
    'HLT_PAL1DoubleMuOpen_v1', 
    'HLT_PAL1ETM30_v1', 
    'HLT_PAL1ETM70_v1', 
    'HLT_PAL1SingleEG12_v1', 
    'HLT_PAL1SingleEG5_v1', 
    'HLT_PAL1SingleEG7_v1', 
    'HLT_PAL1SingleJet16_v1', 
    'HLT_PAL1SingleJet36_v1', 
    'HLT_PAL1SingleMu12_v1', 
    'HLT_PAL1SingleMu3_v1', 
    'HLT_PAL1SingleMu7_v1', 
    'HLT_PAL1SingleMuOpen_v1', 
    'HLT_PAL1Tech53_MB_SingleTrack_v1', 
    'HLT_PAL1Tech53_MB_v1', 
    'HLT_PAL1Tech_HBHEHO_totalOR_v1', 
    'HLT_PAL2DoubleMu3_v1', 
    'HLT_PAMinBiasBSC_OR_v1', 
    'HLT_PAMinBiasBSC_v1', 
    'HLT_PAMinBiasHF_OR_v1', 
    'HLT_PAMinBiasHF_v1', 
    'HLT_PAMinBiasHfOrBSC_v1', 
    'HLT_PAMu12_v1', 
    'HLT_PAMu1_Track1_Mgt2_v1', 
    'HLT_PAMu3PFJet20_v1', 
    'HLT_PAMu3PFJet40_v1', 
    'HLT_PAMu3_Track1_Mgt2_v1', 
    'HLT_PAMu3_v1', 
    'HLT_PAMu5_v1', 
    'HLT_PAMu7PFJet20_v1', 
    'HLT_PAMu7_v1', 
    'HLT_PAMu8_v1', 
    'HLT_PAPhoton10_CaloIdVL_v1', 
    'HLT_PAPhoton15_CaloIdVL_v1', 
    'HLT_PAPhoton20_CaloIdVL_v1', 
    'HLT_PAPhoton30_CaloIdVL_v1', 
    'HLT_PAPixelTracks_Multiplicity100_v1', 
    'HLT_PAPixelTracks_Multiplicity130_v1', 
    'HLT_PAPixelTracks_Multiplicity160_v1', 
    'HLT_PAPixelTracks_Multiplicity190_v1', 
    'HLT_PAPixelTracks_Multiplicity220_v1', 
    'HLT_PAPixelTracks_Multiplicity70_v1', 
    'HLT_PAPixelTracks_Multiplicity90_v1', 
    'HLT_PARandom_v1', 
    'HLT_PARomanPots_Tech52_v1', 
    'HLT_PASingleForJet15_v1', 
    'HLT_PASingleForJet25_v1', 
    'HLT_PAT1minbias_Tech55_v1', 
    'HLT_PAZeroBiasPixel_DoubleTrack_v1', 
    'HLT_PAZeroBiasPixel_SingleTrack_v1', 
    'HLT_PAak5CaloJet20_NoJetID_v1', 
    'HLT_PAak5CaloJet40_NoJetID_v1', 
    'HLT_PAak5CaloJet60_NoJetID_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAZeroBias1_selector
streamA_datasetPAZeroBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAZeroBias1_selector.l1tResults = cms.InputTag('')
streamA_datasetPAZeroBias1_selector.throw      = cms.bool(False)
streamA_datasetPAZeroBias1_selector.triggerConditions = cms.vstring('HLT_PARandom_v1', 
    'HLT_PAZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetPAZeroBias2_selector
streamA_datasetPAZeroBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetPAZeroBias2_selector.l1tResults = cms.InputTag('')
streamA_datasetPAZeroBias2_selector.throw      = cms.bool(False)
streamA_datasetPAZeroBias2_selector.triggerConditions = cms.vstring('HLT_PAL1Tech54_ZeroBias_v1', 
    'HLT_PARandom_v1')

