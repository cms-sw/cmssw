from HLTriggerOffline.SUSYBSM.SUSYBSM_HT_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_MUON_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux350_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux600_cff import *

HLTSusyExoValSeq = cms.Sequence(SUSY_HLT_HT_MET +
                                SUSY_HLT_InclusiveHT +
                                SUSY_HLT_InclusiveMET +
                                SUSY_HLT_MET_BTAG +
                                SUSY_HLT_MET_MUON +
                                SUSY_HLT_InclusiveHT_aux350 + 
                                SUSY_HLT_InclusiveHT_aux600)

HLTSusyExoValSeq_FastSim = cms.Sequence(SUSY_HLT_HT_MET_FASTSIM + 
                                        SUSY_HLT_InclusiveHT_FASTSIM + 
                                        SUSY_HLT_InclusiveMET_FASTSIM + 
                                        SUSY_HLT_MET_BTAG_FASTSIM +
                                        SUSY_HLT_MET_MUON_FASTSIM +
                                        SUSY_HLT_InclusiveHT_aux350 + 
                                        SUSY_HLT_InclusiveHT_aux600)

