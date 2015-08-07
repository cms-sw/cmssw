from HLTriggerOffline.SUSYBSM.SUSYBSM_HT_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_MUON_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux200_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux250_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux300_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux350_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux400_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux475_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux600_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_aux800_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Mu_HT_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Mu_HT_MET_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Mu_HT_BTag_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Mu_HT_Control_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Ele_HT_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Ele_HT_MET_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Ele_HT_BTag_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Ele_HT_Control_SingleLepton_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_MUON_ER_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_HT_MUON_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_HT_MUON_ER_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_HT_MUON_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_Razor_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_caloHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_PhotonHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_HT_DoubleMuon_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_HT_DoubleElectron_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_HT_MuEle_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_Muon_BJet_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_Electron_BJet_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_alphaT_cff import *


SusyExoPostVal = cms.Sequence(SUSY_HLT_HT_MET_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_POSTPROCESSING +
                              SUSY_HLT_InclusiveMET_POSTPROCESSING +
                              SUSY_HLT_InclusiveMET_NoNoiseCleaning_POSTPROCESSING +
                              SUSY_HLT_InclusiveMET_HBHECleaned_POSTPROCESSING +
                              SUSY_HLT_InclusiveMET_JetIdCleaned_POSTPROCESSING +
                              SUSY_HLT_MET_BTAG_POSTPROCESSING + 
                              SUSY_HLT_MET_MUON_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux200_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux250_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux300_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux350_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux400_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux475_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux600_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_aux800_POSTPROCESSING +
                              SUSY_HLT_MET_MUON_ER_POSTPROCESSING +
                              SUSY_HLT_MET_HT_MUON_POSTPROCESSING +
                              SUSY_HLT_MET_HT_MUON_ER_POSTPROCESSING +
                              SUSY_HLT_MET_HT_MUON_BTAG_POSTPROCESSING + 
                              SUSY_HLT_Razor_PostVal_POSTPROCESSING + 
                              SUSY_HLT_CaloHT_POSTPROCESSING + 
                              SUSY_HLT_PhotonHT_POSTPROCESSING +                             
                              SUSY_HLT_HT_DoubleMuon_POSTPROCESSING +
                              SUSY_HLT_HT_DoubleEle_POSTPROCESSING +
                              SUSY_HLT_HT_MuEle_POSTPROCESSING +
                              SUSY_HLT_Muon_BJet_POSTPROCESSING +
                              SUSY_HLT_Electron_BJet_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_MET_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_BTag_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_Control_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_MET_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_BTag_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_Control_SingleLepton_POSTPROCESSING+
                              SUSY_HLT_alphaT_POSTPROCESSING)

SusyExoPostVal_fastsim = cms.Sequence(SUSY_HLT_HT_MET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveMET_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_BTAG_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_MUON_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Mu_HT_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Mu_HT_MET_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Mu_HT_BTag_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Mu_HT_Control_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Ele_HT_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Ele_HT_MET_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Ele_HT_BTag_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Ele_HT_Control_SingleLepton_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux200_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux250_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux300_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux350_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux400_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux475_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux600_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_InclusiveHT_aux800_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_MUON_ER_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_HT_MUON_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_HT_MUON_ER_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_MET_HT_MUON_BTAG_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Razor_PostVal_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_CaloHT_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_PhotonHT_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_HT_DoubleMuon_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_HT_DoubleEle_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_HT_MuEle_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Muon_BJet_FASTSIM_POSTPROCESSING +
                                      SUSY_HLT_Electron_BJet_FASTSIM_POSTPROCESSING)

