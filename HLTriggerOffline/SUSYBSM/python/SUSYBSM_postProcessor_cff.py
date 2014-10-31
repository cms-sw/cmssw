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

SusyExoPostVal = cms.Sequence(SUSY_HLT_HT_MET_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_POSTPROCESSING +
                              SUSY_HLT_InclusiveMET_POSTPROCESSING +
                              SUSY_HLT_MET_BTAG_POSTPROCESSING + 
                              SUSY_HLT_MET_MUON_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_MET_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_BTag_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_MET_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_BTag_SingleLepton_POSTPROCESSING)

SusyExoPostVal_fastsim = cms.Sequence(SUSY_HLT_HT_MET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveMET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_BTAG_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_MUON_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING +
                                      SUSY_HLT_Mu_HT_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Mu_HT_MET_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Mu_HT_BTag_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Ele_HT_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Ele_HT_MET_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Ele_HT_BTag_SingleLepton_FASTSIM_POSTPROCESSING)
