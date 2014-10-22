from HLTriggerOffline.SUSYBSM.SUSYBSM_HT_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_MUON_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_alphaT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux350_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux600_cff import *

SusyExoPostVal = cms.Sequence(SUSY_HLT_HT_MET_POSTPROCESSING +
                             SUSY_HLT_InclusiveHT_POSTPROCESSING +
                             SUSY_HLT_InclusiveMET_POSTPROCESSING +
                             SUSY_HLT_MET_BTAG_POSTPROCESSING + 
                             SUSY_HLT_MET_MUON_POSTPROCESSING +
                             SUSY_HLT_alphaT_POSTPROCESSING +
                             SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING +
                             SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING)

SusyExoPostVal_fastsim = cms.Sequence(SUSY_HLT_HT_MET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveMET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_BTAG_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_MUON_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING)

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
