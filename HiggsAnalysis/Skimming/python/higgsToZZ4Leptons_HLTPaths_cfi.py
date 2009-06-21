import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
import HLTrigger.HLTfilters.hltHighLevel_cfi
higgsToZZ4LeptonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
higgsToZZ4LeptonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLTreprocess")
#Summer08
higgsToZZ4LeptonsHLTFilter.HLTPaths = cms.vstring('HLT_LooseIsoEle15_LW_L1R', 'HLT_DoubleEle10_LW_OnlyPixelM_L1R', 'HLT_IsoMu9', 'HLT_DoubleMu3')
##STARTUP 31X
#higgsToZZ4LeptonsHLTFilter.HLTPaths = cms.vstring('HLT_Ele10_LW_EleId_L1R', 'HLT_DoubleEle5_SW__L1R', 'HLT_Mu9', 'HLT_DoubleMu3')
##IDEAL 31X  
#higgsToZZ4LeptonsHLTFilter.HLTPaths = cms.vstring('HLT_Ele15_SW_LooseTrackIso_L1R', 'HLT_DoubleEle10_SW_L1R', 'HLT_IsoMu9', 'HLT_DoubleMu3')
higgsToZZ4LeptonsHLTFilter.andOr = cms.bool(True)

