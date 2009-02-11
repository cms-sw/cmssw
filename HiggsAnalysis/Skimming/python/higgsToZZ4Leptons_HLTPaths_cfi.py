import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
import HLTrigger.HLTfilters.hltHighLevel_cfi
higgsToZZ4LeptonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
higgsToZZ4LeptonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLTreprocess")
higgsToZZ4LeptonsHLTFilter.HLTPaths = cms.vstring('HLT_IsoMu11', 'HLT_DoubleMu3', 'HLT_IsoEle15_L1I', 'HLT_IsoEle18_L1R', 'HLT_DoubleIsoEle10_L1I', 'HLT_DoubleIsoEle12_L1R')
higgsToZZ4LeptonsHLTFilter.andOr = cms.bool(True)

