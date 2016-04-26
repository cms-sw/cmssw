from HLTriggerOffline.SUSYBSM.SUSYBSM_HT_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_BTAG_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveHT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_inclusiveMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MET_MUON_cff import *
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
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_PhotonMET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_HT_DoubleMuon_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_HT_DoubleElectron_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_HT_MuEle_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_Muon_BJet_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_Electron_BJet_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_alphaT_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MuonFakes_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_ElecFakes_cff import *


HLTSusyExoValSeq = cms.Sequence(SUSY_HLT_HT_MET +
                                SUSY_HLT_InclusiveHT + 
                                SUSY_HLT_InclusiveMET +
                                SUSY_HLT_MET_BTAG +
                                SUSY_HLT_MET_MUON +
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
                                SUSY_HLT_RazorHbb_Rsq0p02_MR300_2CSV0p7_0p4 +
                                SUSY_HLT_RazorHbb_Rsq0p02_MR300_2CSV0p7 +
                                SUSY_HLT_Razor_Main_RsqMR300 + 
                                SUSY_HLT_Razor_QuadJet_RsqMR300 +
                                SUSY_HLT_Razor_DM_Rsq0p36 + 
                                SUSY_HLT_Razor_Main_RsqMR270 + 
                                SUSY_HLT_Razor_QuadJet_RsqMR270 +
                                SUSY_HLT_Razor_DM_Rsq0p30 + 
                                SUSY_HLT_Razor_Main_RsqMR260 + 
                                SUSY_HLT_Razor_QuadJet_RsqMR260 +
                                SUSY_HLT_Razor_Main_RsqMR240 + 
                                SUSY_HLT_Razor_QuadJet_RsqMR240 +
                                SUSY_HLT_Razor_DM_Rsq0p25 + 
                                SUSY_HLT_Razor_Main_Calo_RsqMR240 +
                                SUSY_HLT_Razor_QuadJet_Calo_RsqMR240 +
                                SUSY_HLT_Razor_DM_Calo_Rsq0p25 +
                                SUSY_HLT_CaloHT200 +
                                SUSY_HLT_CaloHT250 +
                                SUSY_HLT_CaloHT300 +
                                SUSY_HLT_CaloHT350 +
                                SUSY_HLT_CaloHT400 +
                                SUSY_HLT_PhotonHT +
                                SUSY_HLT_PhotonMET_pt36 +
                                SUSY_HLT_PhotonMET_pt50 +
                                SUSY_HLT_PhotonMET_pt75 +
                                SUSY_HLT_HT_DoubleMuon +
                                SUSY_HLT_HT_DoubleEle +
                                SUSY_HLT_HT_MuEle +
								SUSY_HLT_HT250_DoubleMuon +
                                SUSY_HLT_HT250_DoubleEle +
                                SUSY_HLT_HT250_MuEle +
                                SUSY_HLT_Muon_BJet +
                                SUSY_HLT_Electron_BJet +
                                SUSY_HLT_HT200_alphaT0p51 +
                                SUSY_HLT_HT200_alphaT0p57 +
                                SUSY_HLT_HT250_alphaT0p55 +
                                SUSY_HLT_HT300_alphaT0p53 +
                                SUSY_HLT_HT350_alphaT0p52 +
                                SUSY_HLT_HT400_alphaT0p51 +
                                SUSY_HLT_HT200_alphaT0p63 +
                                SUSY_HLT_HT250_alphaT0p58 +
                                SUSY_HLT_HT300_alphaT0p54 +
                                SUSY_HLT_HT350_alphaT0p53 +
                                SUSY_HLT_HT400_alphaT0p52 +
                                SUSY_HLT_ElecFakes +
                                SUSY_HLT_MuonFakes
                                )
