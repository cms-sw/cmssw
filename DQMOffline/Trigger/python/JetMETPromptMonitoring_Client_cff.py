import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
pfjetEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_pfjetpT          'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjetpT_numerator          pfjetpT_denominator",
        "effic_pfjetpT_pTThresh 'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_numerator pfjetpT_pTThresh_denominator",
        "effic_pfjetphi       'Jet efficiency vs #phi; PF Jet #phi [rad]; efficiency' pfjetphi_numerator       pfjetphi_denominator",
        "effic_pfjeteta       'Jet efficiency vs #eta; PF Jet #eta; efficiency' pfjeteta_numerator       pfjeteta_denominator",
        ## HB
        "effic_pfjetpT_HB          'Jet pT turnON (HB);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HB_numerator          pfjetpT_HB_denominator",
        "effic_pfjetpT_HB_pTThresh 'Jet pT turnON (HB);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HB_numerator pfjetpT_pTThresh_HB_denominator",
        "effic_pfjetphi_HB       'Jet efficiency vs #phi (HB); PF Jet #phi [rad]; efficiency' pfjetphi_HB_numerator       pfjetphi_HB_denominator",
        "effic_pfjeteta_HB       'Jet efficiency vs #eta (HB); PF Jet #eta; efficiency' pfjeteta_HB_numerator       pfjeteta_HB_denominator",
        ## HE
        "effic_pfjetpT_HE          'Jet pT turnON (HE);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_numerator          pfjetpT_HE_denominator",
        "effic_pfjetpT_HE_pTThresh 'Jet pT turnON (HE);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HE_numerator pfjetpT_pTThresh_HE_denominator",
        "effic_pfjetphi_HE       'Jet efficiency vs #phi (HE); PF Jet #phi [rad]; efficiency' pfjetphi_HE_numerator       pfjetphi_HE_denominator",
        "effic_pfjeteta_HE       'Jet efficiency vs #eta (HE); PF Jet #eta; efficiency' pfjeteta_HE_numerator       pfjeteta_HE_denominator",
         ## HE_p
        "effic_pfjetpT_HE_p          'Jet pT turnON (HEP);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_p_numerator          pfjetpT_HE_p_denominator",
        "effic_pfjetpT_HE_p_pTThresh 'Jet pT turnON (HEP);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HE_p_numerator pfjetpT_pTThresh_HE_p_denominator",
        "effic_pfjetphi_HE_p       'Jet efficiency vs #phi (HEP); PF Jet #phi [rad]; efficiency' pfjetphi_HE_p_numerator       pfjetphi_HE_p_denominator",
        "effic_pfjeteta_HE_p       'Jet efficiency vs #eta (HEP); PF Jet #eta; efficiency' pfjeteta_HE_p_numerator       pfjeteta_HE_p_denominator",
        ## HE_m
        "effic_pfjetpT_HE_m          'Jet pT turnON (HEM);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_m_numerator          pfjetpT_HE_m_denominator",
        "effic_pfjetpT_HE_m_pTThresh 'Jet pT turnON (HEM);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HE_m_numerator pfjetpT_pTThresh_HE_m_denominator",
        "effic_pfjetphi_HE_m       'Jet efficiency vs #phi (HEM); PF Jet #phi [rad]; efficiency' pfjetphi_HE_m_numerator       pfjetphi_HE_m_denominator",
        "effic_pfjeteta_HE_m       'Jet efficiency vs #eta (HEM); PF Jet #eta; efficiency' pfjeteta_HE_m_numerator       pfjeteta_HE_m_denominator",            
        ## HF
        "effic_pfjetpT_HF          'Jet pT turnON (HF);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HF_numerator          pfjetpT_HF_denominator",
        "effic_pfjetpT_HF_pTThresh 'Jet pT turnON (HF);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HF_numerator pfjetpT_pTThresh_HF_denominator",
        "effic_pfjetphi_HF       'Jet efficiency vs #phi (HF); PF Jet #phi [rad]; efficiency' pfjetphi_HF_numerator       pfjetphi_HF_denominator",
        "effic_pfjeteta_HF       'Jet efficiency vs #eta (HF); PF Jet #eta; efficiency' pfjeteta_HF_numerator       pfjeteta_HF_denominator",
        ## 2D Eff
        "effic_pfjetEtaVsPhi       'Jet efficiency vs #eta and #phi; PF Jet #eta; #phi' pfjetEtaVsPhi_numerator       pfjetEtaVsPhi_denominator",
        "effic_pfjetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB); PF Jet #eta; #phi' pfjetEtaVsPhi_HB_numerator       pfjetEtaVsPhi_HB_denominator",
        "effic_pfjetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_numerator       pfjetEtaVsPhi_HE_denominator",
        "effic_pfjetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF); PF Jet #eta; #phi' pfjetEtaVsPhi_HF_numerator       pfjetEtaVsPhi_HF_denominator",

        "effic_pfjetEtaVsPhi_HE_p 'Jet efficiency vs #eta and #phi(HE_p); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_p_numerator       pfjetEtaVsPhi_HE_p_denominator",
        "effic_pfjetEtaVsPhi_HE_m 'Jet efficiency vs #eta and #phi(HE_m); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_m_numerator       pfjetEtaVsPhi_HE_m_denominator",
         
        "effic_pfjetEtaVspT        'Jet efficiency #eta vs Pt;     PF Jet #eta; Pt' pfjetEtaVspT_numerator          pfjetEtaVspT_denominator",
        "effic_pfjetEtaVspT_HB     'Jet efficiency #eta vs Pt(HB); PF Jet #eta; Pt' pfjetEtaVspT_HB_numerator       pfjetEtaVspT_HB_denominator",
        "effic_pfjetEtaVspT_HE     'Jet efficiency #eta vs Pt(HE); PF Jet #eta; Pt' pfjetEtaVspT_HE_numerator       pfjetEtaVspT_HE_denominator",
        "effic_pfjetEtaVspT_HF     'Jet efficiency #eta vs Pt(HF); PF Jet #eta; Pt' pfjetEtaVspT_HF_numerator       pfjetEtaVspT_HF_denominator",

        "effic_pfjetEtaVspT_HE_p   'Jet efficiency #eta vs Pt(HE_p); PF Jet #eta; Pt' pfjetEtaVspT_HE_p_numerator       pfjetEtaVspT_HE_p_denominator",
        "effic_pfjetEtaVspT_HE_m   'Jet efficiency #eta vs Pt(HE_m); PF Jet #eta; Pt' pfjetEtaVspT_HE_m_numerator       pfjetEtaVspT_HE_m_denominator"
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_pfjetpT_vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_numerator pfjetpTVsLS_denominator",
#        "effic_pfjetpT_HBvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HB_numerator pfjetpTVsLS_HB_denominator",
#        "effic_pfjetpT_HEvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HE_numerator pfjetpTVsLS_HE_denominator",
#        "effic_pfjetpT_HFvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HF_numerator pfjetpTVsLS_HF_denominator",
#        "effic_pfjetpT_HE_mvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HE_m_numerator pfjetpTVsLS_HE_m_denominator",
#        "effic_pfjetpT_HE_pvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HE_p_numerator pfjetpTVsLS_HE_p_denominator",
    ),
  
)

calojetEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/*"),
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

        "effic_calojetpT_HF          'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_HF_numerator          calojetpT_HF_denominator",
        "effic_calojetpT_HF_pTThresh 'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HF_numerator calojetpT_pTThresh_HF_denominator",
        "effic_calojetphi_HF       'Jet efficiency vs #phi; Calo Jet #phi [rad]; efficiency' calojetphi_HF_numerator       calojetphi_HF_denominator",
        "effic_calojeteta_HF       'Jet efficiency vs #eta; Calo Jet #eta; efficiency' calojeteta_HF_numerator       calojeteta_HF_denominator",

        ## 2D Eff
        "effic_calojetEtaVsPhi       'Jet efficiency vs #eta and #phi;      Calo Jet #eta; #phi' calojetEtaVsPhi_numerator          calojetEtaVsPhi_denominator",
        "effic_calojetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB);  Calo Jet #eta; #phi' calojetEtaVsPhi_HB_numerator       calojetEtaVsPhi_HB_denominator",
        "effic_calojetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE);  Calo Jet #eta; #phi' calojetEtaVsPhi_HE_numerator       calojetEtaVsPhi_HE_denominator",
        "effic_calojetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF);  Calo Jet #eta; #phi' calojetEtaVsPhi_HF_numerator       calojetEtaVsPhi_HF_denominator",
        "effic_calojetEtaVsPhi_HE_p 'Jet efficiency vs #eta and #phi(HE_p); Calo Jet #eta; #phi' calojetEtaVsPhi_HE_p_numerator     calojetEtaVsPhi_HE_p_denominator",
        "effic_calojetEtaVsPhi_HE_m 'Jet efficiency vs #eta and #phi(HE_m); Calo Jet #eta; #phi' calojetEtaVsPhi_HE_m_numerator     calojetEtaVsPhi_HE_m_denominator",
        
        "effic_calojetEtaVspT        'Jet efficiency #eta vs Pt;        Calo Jet #eta; Pt' calojetEtaVspT_numerator          calojetEtaVspT_denominator",
        "effic_calojetEtaVspT_HB     'Jet efficiency #eta vs Pt(HB);    Calo Jet #eta; Pt' calojetEtaVspT_HB_numerator       calojetEtaVspT_HB_denominator",
        "effic_calojetEtaVspT_HE     'Jet efficiency #eta vs Pt(HE);    Calo Jet #eta; Pt' calojetEtaVspT_HE_numerator       calojetEtaVspT_HE_denominator",
        "effic_calojetEtaVspT_HF     'Jet efficiency #eta vs Pt(HF);    Calo Jet #eta; Pt' calojetEtaVspT_HF_numerator       calojetEtaVspT_HF_denominator",

        "effic_calojetEtaVspT_HE_p   'Jet efficiency #eta vs Pt(HE_p);  Calo Jet #eta; Pt' calojetEtaVspT_HE_p_numerator     calojetEtaVspT_HE_p_denominator",
        "effic_calojetEtaVspT_HE_m   'Jet efficiency #eta vs Pt(HE_m);  Calo Jet #eta; Pt' calojetEtaVspT_HE_m_numerator     calojetEtaVspT_HE_m_denominator"

    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_calojetpT_vs_LS 'JET efficiency vs LS; LS; Calo Jet efficiency' calojetpTVsLS_numerator calojetpTVsLS_denominator",
#        "effic_calojetpT_vs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_numerator calojetpTVsLS_denominator",
#        "effic_calojetpT_HBvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HB_numerator calojetpTVsLS_HB_denominator",
#        "effic_calojetpT_HEvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_numerator calojetpTVsLS_HE_denominator",
#        "effic_calojetpT_HFvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HF_numerator calojetpTVsLS_HF_denominator",
#        "effic_calojetpT_HE_mvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_m_numerator calojetpTVsLS_HE_m_denominator",
#        "effic_calojetpT_HE_pvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_p_numerator calojetpTVsLS_HE_p_denominator",
    ),
  
)



JetMetPromClient = cms.Sequence(
    pfjetEfficiency
    *calojetEfficiency
)
