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


HLTSusyExoValSeq = cms.Sequence(SUSY_HLT_HT_MET +
                                SUSY_HLT_InclusiveHT +
                                SUSY_HLT_InclusiveMET +
                                SUSY_HLT_MET_BTAG +
                                SUSY_HLT_MET_MUON +
                                SUSY_HLT_InclusiveHT_aux350 + 
                                SUSY_HLT_InclusiveHT_aux600 +
                                SUSY_HLT_Mu_HT_SingleLepton +
                                SUSY_HLT_Mu_HT_MET_SingleLepton +
                                SUSY_HLT_Mu_HT_BTag_SingleLepton +
                                SUSY_HLT_Mu_HT_Control_SingleLepton +
                                SUSY_HLT_Ele_HT_SingleLepton +
                                SUSY_HLT_Ele_HT_MET_SingleLepton +
                                SUSY_HLT_Ele_HT_BTag_SingleLepton +
                                SUSY_HLT_Ele_HT_Control_SingleLepton +
                                SUSY_HLT_MET_MUON_ER +
                                SUSY_HLT_MET_HT_MUON +
                                SUSY_HLT_MET_HT_MUON_ER +
                                SUSY_HLT_MET_HT_MUON_BTAG +
                                SUSY_HLT_Razor_Main + 
                                SUSY_HLT_Razor_QuadJet +
                                SUSY_HLT_Razor_DM + 
                                SUSY_HLT_CaloHT200 +
                                SUSY_HLT_CaloHT250 +
                                SUSY_HLT_CaloHT300 +
                                SUSY_HLT_CaloHT350 +
                                SUSY_HLT_CaloHT400 +
                                SUSY_HLT_PhotonHT +
                                SUSY_HLT_HT_DoubleMuon +
                                SUSY_HLT_HT_DoubleEle +
                                SUSY_HLT_HT_MuEle +
                                SUSY_HLT_Muon_BJet +
                                SUSY_HLT_Electron_BJet +
                                SUSY_HLT_HT200_alphaT0p57 +
                                SUSY_HLT_HT250_alphaT0p55 +
                                SUSY_HLT_HT300_alphaT0p53 +
                                SUSY_HLT_HT350_alphaT0p52 +
                                SUSY_HLT_HT400_alphaT0p51 
                                )


HLTSusyExoValSeq_FastSim = cms.Sequence(SUSY_HLT_HT_MET_FASTSIM + 
                                        SUSY_HLT_InclusiveHT_FASTSIM + 
                                        SUSY_HLT_InclusiveMET_FASTSIM + 
                                        SUSY_HLT_MET_BTAG_FASTSIM +
                                        SUSY_HLT_MET_MUON_FASTSIM +
                                        SUSY_HLT_Mu_HT_SingleLepton_FASTSIM +
                                        SUSY_HLT_Mu_HT_MET_SingleLepton_FASTSIM +
                                        SUSY_HLT_Mu_HT_BTag_SingleLepton_FASTSIM +
                                        SUSY_HLT_Mu_HT_Control_SingleLepton_FASTSIM +
                                        SUSY_HLT_Ele_HT_SingleLepton_FASTSIM +
                                        SUSY_HLT_Ele_HT_MET_SingleLepton_FASTSIM +
                                        SUSY_HLT_Ele_HT_BTag_SingleLepton_FASTSIM +
                                        SUSY_HLT_Ele_HT_Control_SingleLepton_FASTSIM +
                                        SUSY_HLT_InclusiveHT_aux350_FASTSIM + 
                                        SUSY_HLT_InclusiveHT_aux600_FASTSIM +
                                        SUSY_HLT_MET_MUON_ER_FASTSIM +
                                        SUSY_HLT_MET_HT_MUON_FASTSIM +
                                        SUSY_HLT_MET_HT_MUON_ER_FASTSIM +
                                        SUSY_HLT_MET_HT_MUON_BTAG_FASTSIM +   
                                        SUSY_HLT_Razor_Main_FASTSIM + 
                                        SUSY_HLT_Razor_QuadJet_FASTSIM +
                                        SUSY_HLT_Razor_DM_FASTSIM + 
                                        SUSY_HLT_CaloHT200_FASTSIM +
                                        SUSY_HLT_CaloHT250_FASTSIM +
                                        SUSY_HLT_CaloHT300_FASTSIM +
                                        SUSY_HLT_CaloHT350_FASTSIM +
                                        SUSY_HLT_CaloHT400_FASTSIM +
                                        SUSY_HLT_PhotonHT_FASTSIM +
                                        SUSY_HLT_HT_DoubleMuon_FASTSIM +
                                        SUSY_HLT_HT_DoubleEle_FASTSIM +
                                        SUSY_HLT_HT_MuEle_FASTSIM +
                                        SUSY_HLT_Muon_BJet_FASTSIM +
                                        SUSY_HLT_Electron_BJet_FASTSIM)

