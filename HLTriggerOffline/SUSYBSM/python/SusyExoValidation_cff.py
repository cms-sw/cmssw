from HLTriggerOffline.SUSYBSM.SUSYBSM_triggerValidation_fastSim_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HT_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_alphaT_cff import *

HLTSusyExoValSeq = cms.Sequence(SUSY_HLT_HT_MET +
                                SUSY_HLT_InclusiveHT +
                                SUSY_HLT_InclusiveMET +
                                SUSY_HLT_MET_BTAG +
                                SUSY_HLT_HT200_alphaT0p57 +
                                SUSY_HLT_HT250_alphaT0p55 +
                                SUSY_HLT_HT300_alphaT0p53 +
                                SUSY_HLT_HT350_alphaT0p52 +
                                SUSY_HLT_HT400_alphaT0p51 
                                )

HLTSusyExoValSeq_FastSim = cms.Sequence(HLTSusyExoValFastSim)

