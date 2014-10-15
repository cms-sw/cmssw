from HLTriggerOffline.SUSYBSM.SUSYBSM_HT_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_MUON_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux350_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux600_cff import *

SusyExoPostVal = cms.Sequence(SUSY_HLT_HT_MET_POSTPROCESSING +
                             SUSY_HLT_InclusiveHT_POSTPROCESSING +
                             SUSY_HLT_InclusiveMET_POSTPROCESSING +
                             SUSY_HLT_MET_BTAG_POSTPROCESSING + 
                             SUSY_HLT_MET_MUON_POSTPROCESSING +
                             SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING +
                             SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING)

SusyExoPostVal_fastsim = cms.Sequence(SUSY_HLT_HT_MET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveMET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_BTAG_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_MUON_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING)

