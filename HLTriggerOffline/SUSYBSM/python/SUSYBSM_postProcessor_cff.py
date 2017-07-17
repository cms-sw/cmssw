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
from HLTriggerOffline.SUSYBSM.SUSYBSM_DiJet_MET_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_HLT_VBF_Mu_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_MuonFakes_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_ElecFakes_cff import *

SusyExoPostVal = cms.Sequence(SUSY_HLT_HT_MET_POSTPROCESSING +
                              SUSY_HLT_InclusiveHT_POSTPROCESSING +
                              SUSY_HLT_InclusiveMET_POSTPROCESSING +
                              SUSYoHLToMEToBTAGoPOSTPROCESSING + 
                              SUSY_HLT_MET_MUON_POSTPROCESSING +
                              SUSYoHLToMEToMUONoERoPOSTPROCESSING +
                              SUSYoHLToMEToHToMUONoPOSTPROCESSING +
                              SUSYoHLToMEToHToMUONoERoPOSTPROCESSING +
                              SUSYoHLToMEToHToMUONoBTAGoPOSTPROCESSING + 
                              SUSYoHLToRazorPostValPOSTPROCESSING + 
                              SUSYoHLToCaloHToPOSTPROCESSING + 
                              SUSYoHLToPhotonHToPOSTPROCESSING +                             
                              SUSYoHLToPhotonMETpt36oPOSTPROCESSING +
                              SUSYoHLToPhotonMETpt50oPOSTPROCESSING +
                              SUSYoHLToPhotonMETpt75oPOSTPROCESSING +
                              SUSYoHLToHToDoubleMuonPOSTPROCESSING +
                              SUSYoHLToHToDoubleElePOSTPROCESSING +
                              SUSYoHLToHToMuElePOSTPROCESSING +
							  SUSYoHLToHT250oDoubleMuonPOSTPROCESSING +
                              SUSYoHLToHT250oDoubleElePOSTPROCESSING +
                              SUSYoHLToHT250oMuElePOSTPROCESSING +
                              SUSYoHLToMuonBJetPOSTPROCESSING +
                              SUSYoHLToElectronBJetPOSTPROCESSING +
                              SUSY_HLT_Mu_HT_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Mu_HT_MET_SingleLepton_POSTPROCESSING +
                              SUSYoHLToMuHToBTagSingleLeptonPOSTPROCESSING +
                              SUSYoHLToMuHToControlSingleLeptonPOSTPROCESSING +
                              SUSY_HLT_Ele_HT_SingleLepton_POSTPROCESSING +
                              SUSY_HLT_Ele_HT_MET_SingleLepton_POSTPROCESSING +
                              SUSYoHLToEleHToBTagSingleLeptonPOSTPROCESSING +
                              SUSYoHLToEleHToControlSingleLeptonPOSTPROCESSING +
                              SUSYoHLTalphaToPOSTPROCESSING +
                              SUSYoHLToDiJetMEToPOSTPROCESSING +
                              SUSYoHLToEleHToControlSingleLeptonPOSTPROCESSING+
                              SUSYoHLTalphaToPOSTPROCESSING+
                              SUSY_HLT_Mu_VBF_POSTPROCESSING+
                              SUSY_HLT_ElecFakes_POSTPROCESSING+
                              SUSY_HLT_MuonFakes_POSTPROCESSING
                              )
