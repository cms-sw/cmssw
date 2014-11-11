import FWCore.ParameterSet.Config as cms
import HLTriggerOffline.SUSYBSM.SUSYBSM_triggerValidation_cff as dir_source

SusyExoPostVal_fastsim = cms.EDAnalyzer("HltSusyExoPostProcessor",
   subDir = dir_source.HLTSusyExoVal.dirname,
   mc_flag = dir_source.HLTSusyExoVal.mc_flag,
   reco_parametersets = dir_source.HLTSusyExoVal.reco_parametersets,
   mc_parametersets = dir_source.HLTSusyExoVal.mc_parametersets
   )

SusyExoPostVal = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLT/SUSYBSM/HLT_PFHT900_v1",
                                  "HLT/SUSYBSM/HLT_PFHT350_PFMET120_NoiseCleaned_v1",
                                  "HLT/SUSYBSM/HLT_PFMET170_NoiseCleaned_v1",
                                  "HLT/SUSYBSM/HLT_PFMET120_NoiseCleaned_BTagCSV07_v1"
                                 ),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)
