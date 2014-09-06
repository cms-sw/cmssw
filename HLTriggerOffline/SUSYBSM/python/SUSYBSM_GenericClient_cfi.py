import FWCore.ParameterSet.Config as cms

SUSYBSM_GenericClient = cms.EDAnalyzer("DQMGenericClient",
  subDirs = cms.untracked.vstring("HLTriggerOffline/SUSYBSM/HLT_PFHT900_v1",
                                  "HLTriggerOffline/SUSYBSM/HLT_PFHT350_PFMET120_NoiseCleaned_v1",
                                  "HLTriggerOffline/SUSYBSM/HLT_PFMET170_NoiseCleaned_v1",
                                  "HLTriggerOffline/SUSYBSM/HLT_PFMET120_NoiseCleaned_BTagCSV07_v1"
                                 ),
  efficiency = cms.vstring(
    "pfMetTurnOn_eff 'Efficiency vs PFMET' pfMetTurnOn_num pfMetTurnOn_den",
    "pfHTTurnOn_eff 'Efficiency vs PFHT' pfHTTurnOn_num pfHTTurnOn_den"
    ),
  resolution = cms.vstring("")
)
