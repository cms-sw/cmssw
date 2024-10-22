import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.Trigger.HEP17Monitoring_Client_cff import *
from DQMOffline.Trigger.HTMonitoring_Client_cff import *
from DQMOffline.Trigger.METMonitoring_Client_cff import *

photonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/EXO/Photon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_photon         'Photon turnON;            Photon pt [GeV]; efficiency'     photon_pt_numerator          photon_pt_denominator",
        "effic_photon_variable 'Photon turnON;            Photon pt [GeV]; efficiency'     photon_pt_variable_numerator photon_pt_variable_denominator",
        "effic_photonPhi       'efficiency vs phi; Photon phi [rad]; efficiency' photon_phi_numerator       photon_phi_denominator",
        "effic_photonEta       'efficiency vs eta; Photon eta; efficiency' photon_eta_numerator       photon_eta_denominator",
        "effic_photonr9       'efficiency vs r9; Photon r9; efficiency' photon_r9_numerator       photon_r9_denominator",
        "effic_photonhoE       'efficiency vs hoE; Photon hoE; efficiency' photon_hoE_numerator       photon_hoE_denominator",
        "effic_photonEtaPhi       'Photon phi; Photon eta; efficiency' photon_etaphi_numerator       photon_etaphi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_photon_vs_LS 'Photon pt efficiency vs LS; LS; PF MET efficiency' photonVsLS_numerator photonVsLS_denominator"
    ),
)

photonVBF_jetMETEfficiency = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Photon/Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50", "HLT/Photon/Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3"),
    subDirs        = cms.untracked.vstring("HLT/EXO/Photon/Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50", "HLT/EXO/Photon/Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_mjj         'Mjj turnON;            Mjj [GeV]; efficiency'     mjj_numerator          mjj_denominator",
        "effic_detajj      'DEtajj turnON;            DEtajj; efficiency'     detajj_numerator       detajj_denominator",
        "effic_met         'MET turnON;            MET [GeV]; efficiency'     met_numerator          met_denominator"
    )
)

muonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/EXO/Muon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                                                          
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muon         'Muon turnON;            Muon pt [GeV]; efficiency'     muon_pt_numerator          muon_pt_denominator",
        "effic_muon_variable 'Muon turnON;            Muon pt [GeV]; efficiency'     muon_pt_variable_numerator muon_pt_variable_denominator",
        "effic_muonPhi       'efficiency vs phi; Muon phi [rad]; efficiency' muon_phi_numerator       muon_phi_denominator",
        "effic_muonEta       'efficiency vs eta; Muon eta; efficiency' muon_eta_numerator       muon_eta_denominator",
        "effic_muonEtaPhi       'Muon phi; Muon eta; efficiency' muon_etaphi_numerator       muon_etaphi_denominator",
        "effic_muondxy       'efficiency vs dxy; Muon dxy; efficiency' muon_dxy_numerator       muon_dxy_denominator",
        "effic_muondz       'efficiency vs dz; Muon dz; efficiency' muon_dz_numerator       muon_dz_denominator",
        "effic_muonetaVB       'efficiency vs eta; Muon eta; efficiency' muon_eta_variablebinning_numerator       muon_eta_variablebinning_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_muon_vs_LS 'Muon pt efficiency vs LS; LS; PF MET efficiency' muonVsLS_numerator muonVsLS_denominator"
    ),

)

NoBPTXEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/EXO/NoBPTX/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetE          'Calo jet energy turnON;            Jet E [GeV]; Efficiency'     jetE_numerator          jetE_denominator",
        "effic_jetE_variable 'Calo jet energy turnON;            Jet E [GeV]; Efficiency'     jetE_variable_numerator jetE_variable_denominator",
        "effic_jetEta          'Calo jet eta eff;            Jet #eta; Efficiency'     jetEta_numerator          jetEta_denominator",
        "effic_jetPhi          'Calo jet phi eff;            Jet #phi; Efficiency'     jetPhi_numerator          jetPhi_denominator",
        "effic_muonPt          'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_numerator          muonPt_denominator",
        "effic_muonPt_variable 'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_variable_numerator muonPt_variable_denominator",
        "effic_muonEta          'Muon eta eff; DisplacedStandAlone Muon #eta; Efficiency'     muonEta_numerator          muonEta_denominator",
        "effic_muonPhi          'Muon phi eff; DisplacedStandAlone Muon #phi; Efficiency'     muonPhi_numerator          muonPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_jetE_vs_LS 'Calo jet energy efficiency vs LS; LS; Jet p_{T} Efficiency' jetEVsLS_numerator jetEVsLS_denominator",
    ),
)

DiDispStaMuonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/EXO/DiDispStaMuon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muonPt          'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_numerator          muonPt_denominator",
        "effic_muonPtNoDxyCut  'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPtNoDxyCut_numerator  muonPtNoDxyCut_denominator",
        "effic_muonPt_variable 'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_variable_numerator muonPt_variable_denominator",
        "effic_muonEta          'Muon eta eff; DisplacedStandAlone Muon #eta; Efficiency'     muonEta_numerator          muonEta_denominator",
        "effic_muonPhi          'Muon phi eff; DisplacedStandAlone Muon #phi; Efficiency'     muonPhi_numerator          muonPhi_denominator",
        "effic_muonDxy          'Muon dxy eff; DisplacedStandAlone Muon #dxy; Efficiency'     muonDxy_numerator          muonDxy_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_muon_vs_LS 'Muon pt efficiency vs LS; LS; Muon p_{T} Efficiency' muonVsLS_numerator muonVsLS_denominator",
    ),
)


METplusTrackEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/EXO/MET/*"),
    verbose = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution = cms.vstring(),
    efficiency = cms.vstring(
        "effic_met_variable    'MET leg turnON;              CaloMET [GeV]; efficiency'     met_variable_numerator    met_variable_denominator",
        "effic_metPhi          'MET leg efficiency vs phi;   CaloMET phi [rad]; efficiency' metPhi_numerator          metPhi_denominator",
        "effic_muonPt_variable 'Track leg turnON;            Muon p_{T} [GeV]; efficiency'  muonPt_variable_numerator muonPt_variable_denominator",
        "effic_muonEta         'Track leg efficiency vs eta; Muon #eta; efficiency'         muonEta_numerator         muonEta_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS     'MET leg efficiency vs LS; LS; CaloMET leg efficiency' metVsLS_numerator metVsLS_denominator",
        "effic_muonPt_vs_LS 'Track leg efficiency vs LS; LS; Track leg efficiency'  muonPtVsLS_numerator muonPtVsLS_denominator",
    ),

)

DisplacedJet_htEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/EXO/DisplacedJet/HT/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_ht          'HT turnON;            PF HT [GeV]; efficiency'     ht_numerator          ht_denominator",
        "effic_ht_variable 'HT turnON;            PF HT [GeV]; efficiency'     ht_variable_numerator ht_variable_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_ht_vs_LS 'HT efficiency vs LS; LS; PF HT efficiency' htVsLS_numerator htVsLS_denominator"
    ),


)

DisplacedJet_jetEfficiency = DQMEDHarvester("DQMGenericClient",
     subDirs        = cms.untracked.vstring("HLT/EXO/DisplacedJet/Jet/*"),
     verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
     resolution     = cms.vstring(),
     efficiency     = cms.vstring(
         "effic_calojetpT          'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_numerator          calojetpT_denominator",
         "effic_calojetpT_pTThresh 'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_numerator calojetpT_pTThresh_denominator",
         "effic_calojetphi       'Jet efficiency vs #phi; Calo Jet #phi [rad]; efficiency' calojetphi_numerator       calojetphi_denominator",
         "effic_calojeteta       'Jet efficiency vs #eta; Calo Jet #eta; efficiency' calojeteta_numerator       calojeteta_denominator",

         "effic_calojetpT_HB          'Jet pT turnON (HB);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HB_numerator          calojetpT_HB_denominator",
         "effic_calojetpT_HB_pTThresh 'Jet pT turnON (HB);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HB_numerator calojetpT_pTThresh_HB_denominator",
         "effic_calojetphi_HB       'Jet efficiency vs #phi (HB); Calo Jet #phi [rad]; efficiency' calojetphi_HB_numerator       calojetphi_HB_denominator",
         "effic_calojeteta_HB       'Jet efficiency vs #eta (HB); Calo Jet #eta; efficiency' calojeteta_HB_numerator       calojeteta_HB_denominator",

         "effic_calojetpT_HE          'Jet pT turnON (HE);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HE_numerator          calojetpT_HE_denominator",
         "effic_calojetpT_HE_pTThresh 'Jet pT turnON (HE);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HE_numerator calojetpT_pTThresh_HE_denominator",
         "effic_calojetphi_HE       'Jet efficiency vs #phi (HE); Calo Jet #phi [rad]; efficiency' calojetphi_HE_numerator       calojetphi_HE_denominator",
         "effic_calojeteta_HE       'Jet efficiency vs #eta (HE); Calo Jet #eta; efficiency' calojeteta_HE_numerator       calojeteta_HE_denominator",

         "effic_calojetpT_HE_p          'Jet pT turnON (HEP);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HE_p_numerator          calojetpT_HE_p_denominator",
         "effic_calojetpT_HE_p_pTThresh 'Jet pT turnON (HEP);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HE_p_numerator calojetpT_pTThresh_HE_P_denominator",
         "effic_calojetphi_HE_p       'Jet efficiency vs #phi (HEP); Calo Jet #phi [rad]; efficiency' calojetphi_HE_p_numerator       calojetphi_HE_p_denominator",
         "effic_calojeteta_HE_p       'Jet efficiency vs #eta (HEP); Calo Jet #eta; efficiency' calojeteta_HE_p_numerator       calojeteta_HE_p_denominator",

         "effic_calojetpT_HE_m          'Jet pT turnON (HEM);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HE_m_numerator          calojetpT_HE_m_denominator",
         "effic_calojetpT_HE_m_pTThresh 'Jet pT turnON (HEM);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HE_m_numerator calojetpT_pTThresh_HE_m_denominator",
         "effic_calojetphi_HE_m       'Jet efficiency vs #phi (HEM); Calo Jet #phi [rad]; efficiency' calojetphi_HE_m_numerator       calojetphi_HE_m_denominator",
         "effic_calojeteta_HE_m       'Jet efficiency vs #eta (HEM); Calo Jet #eta; efficiency' calojeteta_HE_m_numerator       calojeteta_HE_m_denominator",

         "effic_calojetpT_HEP17          'Jet pT turnON (HEP17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEP17_numerator          calojetpT_HEP17_denominator",
         "effic_calojetpT_HEP17_pTThresh 'Jet pT turnON (HEP17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HEP17_numerator calojetpT_pTThresh_HEP17_denominator",
         "effic_calojetphi_HEP17       'Jet efficiency vs #phi (HEP17); Calo Jet #phi [rad]; efficiency' calojetphi_HEP17_numerator       calojetphi_HEP17_denominator",
         "effic_calojeteta_HEP17       'Jet efficiency vs #eta (HEP17); Calo Jet #eta; efficiency' calojeteta_HEP17_numerator       calojeteta_HEP17_denominator",
         "effic_calojetabseta_HEP17       'Jet efficiency vs |#eta| (HEP17); Calo Jet |#eta|; efficiency' calojetabseta_HEP17_numerator       calojetabseta_HEP17_denominator",

         "effic_calojetpT_HEP18          'Jet pT turnON (HEP18);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEP18_numerator          calojetpT_HEP18_denominator",
         "effic_calojetpT_HEP18_pTThresh 'Jet pT turnON (HEP18);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HEP18_numerator calojetpT_pTThresh_HEP18_denominator",
         "effic_calojetphi_HEP18       'Jet efficiency vs #phi (HEP18); Calo Jet phi [rad]; efficiency' calojetphi_HEP18_numerator       calojetphi_HEP18_denominator",
         "effic_calojeteta_HEP18       'Jet efficiency vs #eta (HEP18); Calo Jet #eta; efficiency' calojeteta_HEP18_numerator       calojeteta_HEP18_denominator",

         "effic_calojetpT_HEM17          'Jet pT turnON (HEM17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEM17_numerator          calojetpT_HEM17_denominator",
         "effic_calojetpT_HEM17_pTThresh 'Jet pT turnON (HEM17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HEM17_numerator calojetpT_pTThresh_HEM17_denominator",
         "effic_calojetphi_HEM17       'Jet efficiency vs #phi (HEM17); Calo Jet #phi; efficiency' calojetphi_HEM17_numerator       calojetphi_HEM17_denominator",
         "effic_calojeteta_HEM17       'Jet efficiency vs #eta (HEM17); Calo Jet #eta; efficiency' calojeteta_HEM17_numerator       calojeteta_HEM17_denominator",
         "effic_calojetabseta_HEM17       'Jet efficiency vs #eta (HEM17); Calo Jet |#eta|; efficiency' calojetabseta_HEM17_numerator       calojetabseta_HEM17_denominator",

         "effic_calojetpT_HF          'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_HF_numerator          calojetpT_HF_denominator",
         "effic_calojetpT_HF_pTThresh 'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HF_numerator calojetpT_pTThresh_HF_denominator",
         "effic_calojetphi_HF       'Jet efficiency vs #phi; Calo Jet #phi [rad]; efficiency' calojetphi_HF_numerator       calojetphi_HF_denominator",
         "effic_calojeteta_HF       'Jet efficiency vs #eta; Calo Jet #eta; efficiency' calojeteta_HF_numerator       calojeteta_HF_denominator",

         ## 2D Eff
         "effic_calojetEtaVsPhi       'Jet efficiency vs #eta and #phi; PF Jet #eta; #phi' calojetEtaVsPhi_numerator       calojetEtaVsPhi_denominator",
         "effic_calojetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB); PF Jet #eta; #phi' calojetEtaVsPhi_HB_numerator       calojetEtaVsPhi_HB_denominator",
         "effic_calojetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE); PF Jet #eta; #phi' calojetEtaVsPhi_HE_numerator       calojetEtaVsPhi_HE_denominator",
         "effic_calojetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF); PF Jet #eta; #phi' calojetEtaVsPhi_HF_numerator       calojetEtaVsPhi_HF_denominator",
         "effic_calojetEtaVsPhi_HEM17 'Jet efficiency vs #eta and #phi(HEM17); PF Jet #eta; #phi' calojetEtaVsPhi_HEM17_numerator       calojetEtaVsPhi_HEM17_denominator",
         "effic_calojetEtaVsPhi_HEP17 'Jet efficiency vs #eta and #phi(HEP17); PF Jet #eta; #phi' calojetEtaVsPhi_HEP17_numerator       calojetEtaVsPhi_HEP17_denominator",
         "effic_calojetEtaVsPhi_HEP18 'Jet efficiency vs #eta and #phi(HEP18); PF Jet #eta; #phi' calojetEtaVsPhi_HEP18_numerator       calojetEtaVsPhi_HEP18_denominator",
         "effic_calojetEtaVsPhi_HE_p 'Jet efficiency vs #eta and #phi(HE_p); PF Jet #eta; #phi' calojetEtaVsPhi_HE_p_numerator       calojetEtaVsPhi_HE_p_denominator",
         "effic_calojetEtaVsPhi_HE_m 'Jet efficiency vs #eta and #phi(HE_m); PF Jet #eta; #phi' calojetEtaVsPhi_HE_m_numerator       calojetEtaVsPhi_HE_m_denominator",
         "effic_calojetAbsEtaVsPhi_HEM17 'Jet efficiency vs |#eta| and #phi(HEM17); PF Jet |#eta|; #phi' calojetAbsEtaVsPhi_HEM17_numerator       calojetAbsEtaVsPhi_HEM17_denominator",
         "effic_calojetAbsEtaVsPhi_HEP17 'Jet efficiency vs |#eta| and #phi(HEP17); PF Jet |#eta|; #phi' calojetAbsEtaVsPhi_HEP17_numerator       calojetAbsEtaVsPhi_HEP17_denominator",
     ),
     efficiencyProfile = cms.untracked.vstring(
         "effic_calojetpT_vs_LS 'JET efficiency vs LS; LS; Calo Jet efficiency' calojetpTVsLS_numerator calojetpTVsLS_denominator",
         "effic_calojetpT_vs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_numerator calojetpTVsLS_denominator",
         "effic_calojetpT_HBvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HB_numerator calojetpTVsLS_HB_denominator",
         "effic_calojetpT_HEvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_numerator calojetpTVsLS_HE_denominator",
         "effic_calojetpT_HFvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HF_numerator calojetpTVsLS_HF_denominator",
         "effic_calojetpT_HEP17vs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HEP17_numerator calojetpTVsLS_HEP17_denominator",
         "effic_calojetpT_HEM17vs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HEM17_numerator calojetpTVsLS_HEM17_denominator",
         "effic_calojetpT_HE_mvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_m_numerator calojetpTVsLS_HE_m_denominator",
         "effic_calojetpT_HE_pvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_p_numerator calojetpTVsLS_HE_p_denominator",
     ),

)

DisplacedJet_jetRatioHemHep17 = DQMEDHarvester("DQMGenericClient",
    subDirs =  cms.untracked.vstring("HLT/EXO/DisplacedJet/Jet/*"),
    verbose = cms.untracked.uint32(0),
    resolution = cms.vstring(),
    efficiency = cms.vstring(
         "ratio_calojetpT_HEP17VSHEM17          'HEP17/HEM17 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17 effic_calojetpT_HEM17 simpleratio" ,
         "ratio_calojetpT_pTTresh_HEP17VSHEM17  'HEP17/HEM17 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17_pTThresh effic_calojetpT_HEM17_pTThresh simpleratio" ,
         "ratio_calojetphi_HEP17VSHEM17         'HEP17/HEM17 vs #phi;          CaloJet #phi [GeV]; Ratio'  effic_calojetphi_HEP17 effic_calojetphi_HEM17 simpleratio",
         "ratio_calojeteta_HEP17VSHEM17         'HEP17/HEM17 vs |#eta|;        CaloJet(|#eta|) ; Ratio'    effic_calojetabseta_HEP17 effic_calojetabseta_HEM17 simpleratio",

         "ratio_calojetpT_HEP17VSHEP18          'HEP17/HEP18 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17 effic_calojetpT_HEP18 simpleratio" ,
         "ratio_calojetpT_pTTresh_HEP17VSHEP18  'HEP17/HEP18 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17_pTThresh effic_calojetpT_HEP18_pTThresh simpleratio" ,
    )


 )

DisplacedVertices_PFHT330_TripleBTag = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/EXO/DisplacedVertices/FullyHadronic/*"),
    verbose        = cms.untracked.uint32(0),
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetPt_1       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_numerator       jetPt_1_denominator",
        "effic_jetPt_2       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_numerator       jetPt_2_denominator",
        "effic_jetPt_3       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_numerator       jetPt_3_denominator",
        "effic_jetPt_4       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_numerator       jetPt_4_denominator",

        "effic_jetEta_1      'efficiency vs 1st jet #eta;jet #eta;efficiency' jetEta_1_numerator     jetEta_1_denominator",
        "effic_jetEta_2      'efficiency vs 2nd jet #eta;jet #eta;efficiency' jetEta_2_numerator     jetEta_2_denominator",
        "effic_jetEta_3      'efficiency vs 3rd jet #eta;jet #eta;efficiency' jetEta_3_numerator     jetEta_3_denominator",
        "effic_jetEta_4      'efficiency vs 4th jet #eta;jet #eta;efficiency' jetEta_4_numerator     jetEta_4_denominator",

        "effic_jetPhi_1      'efficiency vs 1st jet #phi;jet #phi;efficiency'    jetPhi_1_numerator      jetPhi_1_denominator",
        "effic_jetPhi_2      'efficiency vs 2nd jet #phi;jet #phi;efficiency'    jetPhi_2_numerator      jetPhi_2_denominator",
        "effic_jetPhi_3      'efficiency vs 3rd jet #phi;jet #phi;efficiency'    jetPhi_3_numerator      jetPhi_3_denominator",
        "effic_jetPhi_4      'efficiency vs 4th jet #phi;jet #phi;efficiency'    jetPhi_4_numerator      jetPhi_4_denominator",

        "effic_bjetPt_1      'efficiency vs 1st b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_1_numerator  bjetPt_1_denominator",
        "effic_bjetPt_2      'efficiency vs 2nd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_2_numerator  bjetPt_2_denominator",
        "effic_bjetPt_3      'efficiency vs 3rd b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_3_numerator  bjetPt_3_denominator",
        "effic_bjetPt_4      'efficiency vs 4th b-jet p_{T};bjet p_{T} [GeV];efficiency' bjetPt_4_numerator  bjetPt_4_denominator",

        "effic_bjetEta_1     'efficiency vs 1st b-jet #eta;bjet #eta;efficiency'  bjetEta_1_numerator   bjetEta_1_denominator",
        "effic_bjetEta_2     'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency'  bjetEta_2_numerator   bjetEta_2_denominator",
        "effic_bjetEta_3     'efficiency vs 3rd b-jet #eta;bjet #eta;efficiency'  bjetEta_3_numerator   bjetEta_3_denominator",
        "effic_bjetEta_4     'efficiency vs 4th b-jet #eta;bjet #eta;efficiency'  bjetEta_4_numerator   bjetEta_4_denominator",

        "effic_bjetPhi_1     'efficiency vs 1st b-jet #phi;bjet #phi;efficiency'  bjetPhi_1_numerator   bjetPhi_1_denominator",
        "effic_bjetPhi_2     'efficiency vs 2nd b-jet #phi;bjet #phi;efficiency'  bjetPhi_2_numerator   bjetPhi_2_denominator",
        "effic_bjetPhi_3     'efficiency vs 3rd b-jet #phi;bjet #phi;efficiency'  bjetPhi_3_numerator   bjetPhi_3_denominator",
        "effic_bjetPhi_4     'efficiency vs 4th b-jet #phi;bjet #phi;efficiency'  bjetPhi_4_numerator   bjetPhi_4_denominator",

        "effic_bjetCSV_1     'efficiency vs 1st b-jet Discrim;bjet Discrim;efficiency' bjetCSV_1_numerator  bjetCSV_1_denominator",
        "effic_bjetCSV_2     'efficiency vs 2nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_2_numerator  bjetCSV_2_denominator",
        "effic_bjetCSV_3     'efficiency vs 3nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_3_numerator  bjetCSV_3_denominator",
        "effic_bjetCSV_4     'efficiency vs 4nd b-jet Discrim;bjet Discrim;efficiency' bjetCSV_4_numerator  bjetCSV_4_denominator",

        "effic_eventHT       'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_numerator       eventHT_denominator",
        "effic_jetEtaPhi_HEP17       'efficiency vs jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_HEP17_numerator       jetEtaPhi_HEP17_denominator",

        "effic_jetPt_1_variableBinning       'efficiency vs 1st jet p_{T};jet p_{T} [GeV];efficiency' jetPt_1_variableBinning_numerator       jetPt_1_variableBinning_denominator",
        "effic_jetPt_2_variableBinning       'efficiency vs 2nd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_2_variableBinning_numerator       jetPt_2_variableBinning_denominator",
        "effic_jetPt_3_variableBinning       'efficiency vs 3rd jet p_{T};jet p_{T} [GeV];efficiency' jetPt_3_variableBinning_numerator       jetPt_3_variableBinning_denominator",
        "effic_jetPt_4_variableBinning       'efficiency vs 4th jet p_{T};jet p_{T} [GeV];efficiency' jetPt_4_variableBinning_numerator       jetPt_4_variableBinning_denominator",

        "effic_bjetEta_1_variableBinning  'efficiency vs 1st b-jet #eta;bjet #eta;efficiency' bjetEta_1_variableBinning_numerator     bjetEta_1_variableBinning_denominator",
        "effic_bjetEta_2_variableBinning  'efficiency vs 2nd b-jet #eta;bjet #eta;efficiency' bjetEta_2_variableBinning_numerator     bjetEta_2_variableBinning_denominator",
        "effic_bjetEta_3_variableBinning  'efficiency vs 3rd b-jet #eta;bjet #eta;efficiency' bjetEta_3_variableBinning_numerator     bjetEta_3_variableBinning_denominator",
        "effic_bjetEta_4_variableBinning  'efficiency vs 4th b-jet #eta;bjet #eta;efficiency' bjetEta_4_variableBinning_numerator     bjetEta_4_variableBinning_denominator",

        "effic_eventHT_variableBinning    'efficiency vs event H_{T};event H_{T} [GeV];efficiency' eventHT_variableBinning_numerator       eventHT_variableBinning_denominator",

        "effic_jetMulti       'efficiency vs jet multiplicity;jet multiplicity;efficiency' jetMulti_numerator       jetMulti_denominator",
        "effic_bjetMulti      'efficiency vs b-jet multiplicity;bjet multiplicity;efficiency' bjetMulti_numerator   bjetMulti_denominator",

        "effic_jetPtEta_1     'efficiency vs 1st jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_1_numerator       jetPtEta_1_denominator",
        "effic_jetPtEta_2     'efficiency vs 2nd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_2_numerator       jetPtEta_2_denominator",
        "effic_jetPtEta_3     'efficiency vs 3rd jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_3_numerator       jetPtEta_3_denominator",
        "effic_jetPtEta_4     'efficiency vs 4th jet p_{T}-#eta;jet p_{T} [GeV];jet #eta' jetPtEta_4_numerator       jetPtEta_4_denominator",

        "effic_jetEtaPhi_1    'efficiency vs 1st jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_1_numerator       jetEtaPhi_1_denominator",
        "effic_jetEtaPhi_2    'efficiency vs 2nd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_2_numerator       jetEtaPhi_2_denominator",
        "effic_jetEtaPhi_3    'efficiency vs 3rd jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_3_numerator       jetEtaPhi_3_denominator",
        "effic_jetEtaPhi_4    'efficiency vs 4th jet #eta-#phi;jet #eta;jet #phi' jetEtaPhi_4_numerator       jetEtaPhi_4_denominator",

        "effic_bjetPtEta_1    'efficiency vs 1st b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_1_numerator   bjetPtEta_1_denominator",
        "effic_bjetPtEta_2    'efficiency vs 2nd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_2_numerator   bjetPtEta_2_denominator",
        "effic_bjetPtEta_3    'efficiency vs 3rd b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_3_numerator   bjetPtEta_3_denominator",
        "effic_bjetPtEta_4    'efficiency vs 4th b-jet p_{T}-#eta;jet p_{T} [GeV];bjet #eta' bjetPtEta_4_numerator   bjetPtEta_4_denominator",

        "effic_bjetEtaPhi_1    'efficiency vs 1st b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_1_numerator  bjetEtaPhi_1_denominator",
        "effic_bjetEtaPhi_2    'efficiency vs 2nd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_2_numerator  bjetEtaPhi_2_denominator",
        "effic_bjetEtaPhi_3    'efficiency vs 3rd b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_3_numerator  bjetEtaPhi_3_denominator",
        "effic_bjetEtaPhi_4    'efficiency vs 4th b-jet #eta-#phi;bjet #eta;bjet #phi' bjetEtaPhi_4_numerator  bjetEtaPhi_4_denominator",

        "effic_bjetCSVHT_1 'efficiency vs 1st b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_1_numerator bjetCSVHT_1_denominator"
        "effic_bjetCSVHT_2 'efficiency vs 2nd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_2_numerator bjetCSVHT_2_denominator"
        "effic_bjetCSVHT_3 'efficiency vs 3rd b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_3_numerator bjetCSVHT_3_denominator"
        "effic_bjetCSVHT_4 'efficiency vs 4th b-jet Discrim - event H_{T};bjet Discrim;event H_{T} [GeV]' bjetCSVHT_4_numerator bjetCSVHT_4_denominator"
    ),
)

exoticaClient = cms.Sequence(
    NoBPTXEfficiency
  + DiDispStaMuonEfficiency
  + photonEfficiency
  + photonVBF_jetMETEfficiency
  + DisplacedJet_htEfficiency
  + (DisplacedJet_jetEfficiency*DisplacedJet_jetRatioHemHep17)
  + htClient
  + metClient
  + METplusTrackEfficiency
  + muonEfficiency
  + hep17Efficiency
)

from DQMOffline.Trigger.TrackingMonitoring_Client_cff import *

#DisplacedJet Track Monitoring
trackingforDisplacedJetEffFromHitPatternHLT = trackingEffFromHitPatternHLT.clone(
    subDirs = [
    "HLT/EXO/DisplacedJet/Tracking/iter2MergedForBTag/HitEffFromHitPattern*",
    "HLT/EXO/DisplacedJet/Tracking/iter4ForDisplaced/HitEffFromHitPattern*",
]
)
trackingForDisplacedJetMonitorClientHLT  = cms.Sequence(
    trackingforDisplacedJetEffFromHitPatternHLT
)


