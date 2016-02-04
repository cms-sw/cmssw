import FWCore.ParameterSet.Config as cms
import HLTriggerOffline.SUSYBSM.SUSYBSM_triggerValidation_cff as dir_source

SusyExoPostVal = cms.EDAnalyzer("HltSusyExoPostProcessor",
   subDir = dir_source.HLTSusyExoVal.dirname,
   mc_flag = dir_source.HLTSusyExoVal.mc_flag,
   reco_parametersets = dir_source.HLTSusyExoVal.reco_parametersets,
   mc_parametersets = dir_source.HLTSusyExoVal.mc_parametersets
   )
