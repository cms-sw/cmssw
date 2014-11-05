from HLTriggerOffline.SUSYBSM.SUSYBSM_HT_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_MUON_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux350_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux600_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Mu_HT_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Mu_HT_MET_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Mu_HT_BTag_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Ele_HT_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Ele_HT_MET_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Ele_HT_BTag_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_alphaT_cff import *

HLTSusyExoValSeq = cms.Sequence(SUSY_HLT_HT_MET +
                                SUSY_HLT_InclusiveHT +
                                SUSY_HLT_InclusiveMET +
                                SUSY_HLT_MET_BTAG +
                                SUSY_HLT_MET_MUON +
                                SUSY_HLT_HT200_alphaT0p57 +
                                SUSY_HLT_HT250_alphaT0p55 +
                                SUSY_HLT_HT300_alphaT0p53 +
                                SUSY_HLT_HT350_alphaT0p52 +
                                SUSY_HLT_HT400_alphaT0p51 +
                                SUSY_HLT_InclusiveHT_aux350 + 
                                SUSY_HLT_InclusiveHT_aux600 +
                                SUSY_HLT_Mu_HT_SingleLepton +
                                SUSY_HLT_Mu_HT_MET_SingleLepton +
                                SUSY_HLT_Mu_HT_BTag_SingleLepton +
                                SUSY_HLT_Ele_HT_SingleLepton +
                                SUSY_HLT_Ele_HT_MET_SingleLepton +
                                SUSY_HLT_Ele_HT_BTag_SingleLepton)

HLTSusyExoValSeq_FastSim = cms.Sequence(SUSY_HLT_HT_MET_FASTSIM + 
                                        SUSY_HLT_InclusiveHT_FASTSIM + 
                                        SUSY_HLT_InclusiveMET_FASTSIM + 
                                        SUSY_HLT_MET_BTAG_FASTSIM +
                                        SUSY_HLT_MET_MUON_FASTSIM +
                                        SUSY_HLT_InclusiveHT_aux350 + 
                                        SUSY_HLT_InclusiveHT_aux600 +
                                        SUSY_HLT_Mu_HT_SingleLepton_FASTSIM +
                                        SUSY_HLT_Mu_HT_MET_SingleLepton_FASTSIM +
                                        SUSY_HLT_Mu_HT_BTag_SingleLepton_FASTSIM +
                                        SUSY_HLT_Ele_HT_SingleLepton_FASTSIM +
                                        SUSY_HLT_Ele_HT_MET_SingleLepton_FASTSIM +
                                        SUSY_HLT_Ele_HT_BTag_SingleLepton_FASTSIM)

