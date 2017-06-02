import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
pfjetEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JetMET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_pfjetpT          'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjetpT_numerator          pfjetpT_denominator",
        "effic_pfjetpT_variable 'Jet pT turnON;            PFJet(pT) [GeV]; efficiency'     pfjet_variable_numerator pfjet_variable_denominator",
        "effic_pfjetPhi       'Jet efficiency vs #phi; PF Jet #phi [rad]; efficiency' pfjetPhi_numerator       pfjetPhi_denominator",
        "effic_pfjetEta       'Jet efficiency vs #eta; PF Jet #eta [rad]; efficiency' pfjetEta_numerator       pfjetEta_denominator",
        ## HB
        "effic_pfjetpT_HB          'Jet pT turnON (HB);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HB_numerator          pfjetpT_HB_denominator",
        "effic_pfjetpT_HB_variable 'Jet pT turnON (HB);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HB_numerator pfjetpT_variable_HB_denominator",
        "effic_pfjetPhi_HB       'Jet efficiency vs #phi (HB); PF Jet #phi [rad]; efficiency' pfjetPhi_HB_numerator       pfjetPhi_HB_denominator",
        "effic_pfjetEta_HB       'Jet efficiency vs #eta (HB); PF Jet #eta [rad]; efficiency' pfjetEta_HB_numerator       pfjetEta_HB_denominator",

        "effic_pfjetpT_HE          'Jet pT turnON (HE);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_numerator          pfjetpT_HE_denominator",
        "effic_pfjetpT_HE_variable 'Jet pT turnON (HE);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HE_numerator pfjetpT_variable_HE_denominator",
        "effic_pfjetPhi_HE       'Jet efficiency vs #phi (HE); PF Jet #phi [rad]; efficiency' pfjetPhi_HE_numerator       pfjetPhi_HE_denominator",
        "effic_pfjetEta_HE       'Jet efficiency vs #eta (HE); PF Jet #eta [rad]; efficiency' pfjetEta_HE_numerator       pfjetEta_HE_denominator",

        "effic_pfjetpT_HE_p          'Jet pT turnON (HEP);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_p_numerator          pfjetpT_HE_p_denominator",
        "effic_pfjetpT_HE_p_variable 'Jet pT turnON (HEP);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HE_p_numerator pfjetpT_variable_HE_p_denominator",
        "effic_pfjetPhi_HE_p       'Jet efficiency vs #phi (HEP); PF Jet #phi [rad]; efficiency' pfjetPhi_HE_p_numerator       pfjetPhi_HE_p_denominator",
        "effic_pfjetEta_HE_p       'Jet efficiency vs #eta (HEP); PF Jet #eta [rad]; efficiency' pfjetEta_HE_p_numerator       pfjetEta_HE_p_denominator",

        "effic_pfjetpT_HE_m          'Jet pT turnON (HEM);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_m_numerator          pfjetpT_HE_m_denominator",
        "effic_pfjetpT_HE_m_variable 'Jet pT turnON (HEM);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HE_m_numerator pfjetpT_variable_HE_m_denominator",
        "effic_pfjetPhi_HE_m       'Jet efficiency vs #phi (HEM); PF Jet phi [rad]; efficiency' pfjetPhi_HE_m_numerator       pfjetPhi_HE_m_denominator",
        "effic_pfjetEta_HE_m       'Jet efficiency vs #eta (HEM); PF Jet phi [rad]; efficiency' pfjetEta_HE_m_numerator       pfjetEta_HE_m_denominator",

        "effic_pfjetpT_HEP17          'Jet pT turnON (HEP17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HEP17_numerator          pfjetpT_HEP17_denominator",
        "effic_pfjetpT_HEP17_variable 'Jet pT turnON (HEP17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HEP17_numerator pfjetpT_variable_HEP17_denominator",
        "effic_pfjetPhi_HEP17       'Jet efficiency vs #phi (HEP17); PF Jet #phi [rad]; efficiency' pfjetPhi_HEP17_numerator       pfjetPhi_HEP17_denominator",
        "effic_pfjetEta_HEP17       'Jet efficiency vs #eta (HEP17); PF Jet #eta [rad]; efficiency' pfjetEta_HEP17_numerator       pfjetEta_HEP17_denominator",

        "effic_pfjetpT_HEM17          'Jet pT turnON (HEM17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HEM17_numerator          pfjetpT_HEM17_denominator",
        "effic_pfjetpT_HEM17_variable 'Jet pT turnON (HEM17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HEM17_numerator pfjetpT_variable_HEM17_denominator",
        "effic_pfjetPhi_HEM17       'Jet efficiency vs #phi (HEM17); PF Jet phi [rad]; efficiency' pfjetPhi_HEM17_numerator       pfjetPhi_HEM17_denominator",
        "effic_pfjetEta_HEM17       'Jet efficiency vs #eta (HEM17); PF Jet phi [rad]; efficiency' pfjetEta_HEM17_numerator       pfjetEta_HEM17_denominator",

        "effic_pfjetpT_HEP18          'Jet pT turnON (HEP18);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HEP18_numerator          pfjetpT_HEP18_denominator",
        "effic_pfjetpT_HEP18_variable 'Jet pT turnON (HEP18);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HEP18_numerator pfjetpT_variable_HEP18_denominator",
        "effic_pfjetPhi_HEP18       'Jet efficiency vs #phi (HEP18); PF Jet #phi [rad]; efficiency' pfjetPhi_HEP18_numerator       pfjetPhi_HEP18_denominator",
        "effic_pfjetEta_HEP18       'Jet efficiency vs #eta (HEP18); PF Jet #eta [rad]; efficiency' pfjetEta_HEP18_numerator       pfjetEta_HEP18_denominator",

        "effic_pfjetpT_HF          'Jet pT turnON (HF);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HF_numerator          pfjetpT_HF_denominator",
        "effic_pfjetpT_HF_variable 'Jet pT turnON (HF);            PFJet(pT) [GeV]; efficiency'     pfjetpT_variable_HF_numerator pfjetpT_variable_HF_denominator",
        "effic_pfjetPhi_HF       'Jet efficiency vs #phi (HF); PF Jet phi [rad]; efficiency' pfjetPhi_HF_numerator       pfjetPhi_HF_denominator",
        "effic_pfjetEta_HF       'Jet efficiency vs #eta (HF); PF Jet phi [rad]; efficiency' pfjetEta_HF_numerator       pfjetEta_HF_denominator",
        ## 2D Eff
        "effic_pfjetEtaVsPhi       'Jet efficiency vs #eta and #phi; PF Jet #eta; #phi' pfjetEtaVsPhi_numerator       pfjetEtaVsPhi_denominator",
        "effic_pfjetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB); PF Jet #eta; #phi' pfjetEtaVsPhi_HB_numerator       pfjetEtaVsPhi_HB_denominator",
        "effic_pfjetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_numerator       pfjetEtaVsPhi_HE_denominator",
        "effic_pfjetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF); PF Jet #eta; #phi' pfjetEtaVsPhi_HF_numerator       pfjetEtaVsPhi_HF_denominator",
        "effic_pfjetEtaVsPhi_HEM17 'Jet efficiency vs #eta and #phi(HEM17); PF Jet #eta; #phi' pfjetEtaVsPhi_HEM17_numerator       pfjetEtaVsPhi_HEM17_denominator",
        "effic_pfjetEtaVsPhi_HEP17 'Jet efficiency vs #eta and #phi(HEP17); PF Jet #eta; #phi' pfjetEtaVsPhi_HEP17_numerator       pfjetEtaVsPhi_HEP17_denominator",
        "effic_pfjetEtaVsPhi_HEP18 'Jet efficiency vs #eta and #phi(HEP18); PF Jet #eta; #phi' pfjetEtaVsPhi_HEP18_numerator       pfjetEtaVsPhi_HEP18_denominator",
        "effic_pfjetEtaVsPhi_HE_p 'Jet efficiency vs #eta and #phi(HE_p); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_p_numerator       pfjetEtaVsPhi_HE_p_denominator",
        "effic_pfjetEtaVsPhi_HE_m 'Jet efficiency vs #eta and #phi(HE_m); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_m_numerator       pfjetEtaVsPhi_HE_m_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_pfjetpT_vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_numerator pfjetpTVsLS_denominator",
        "effic_pfjetpT_HBvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HB_numerator pfjetpTVsLS_HB_denominator",
        "effic_pfjetpT_HEvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HE_numerator pfjetpTVsLS_HE_denominator",
        "effic_pfjetpT_HFvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HF_numerator pfjetpTVsLS_HF_denominator",
        "effic_pfjetpT_HEP17vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HEP17_numerator pfjetpTVsLS_HEP17_denominator",
        "effic_pfjetpT_HEM17vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HEM17_numerator pfjetpTVsLS_HEM17_denominator",
        "effic_pfjetpT_HE_mvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HE_m_numerator pfjetpTVsLS_HE_m_denominator",
        "effic_pfjetpT_HE_pvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HE_p_numerator pfjetpTVsLS_HE_p_denominator",
    ),
  
)

calojetEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JetMET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_calojetpT          'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_numerator          calojetpT_denominator",
        "effic_calojetpT_variable 'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojet_variable_numerator calojet_variable_denominator",
        "effic_calojetPhi       'Jet efficiency vs #phi; Calo Jet phi [rad]; efficiency' calojetPhi_numerator       calojetPhi_denominator",
        "effic_calojetEta       'Jet efficiency vs #eta; Calo Jet phi [rad]; efficiency' calojetEta_numerator       calojetEta_denominator",

        "effic_calojetpT_HB          'Jet pT turnON (HB);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HB_numerator          calojetpT_HB_denominator",
        "effic_calojetpT_HB_variable 'Jet pT turnON (HB);            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HB_numerator calojetpT_variable_HB_denominator",
        "effic_calojetPhi_HB       'Jet efficiency vs #phi (HB); Calo Jet phi [rad]; efficiency' calojetPhi_HB_numerator       calojetPhi_HB_denominator",
        "effic_calojetEta_HB       'Jet efficiency vs #eta (HB); Calo Jet phi [rad]; efficiency' calojetEta_HB_numerator       calojetEta_HB_denominator",

        "effic_calojetpT_HE          'Jet pT turnON (HE);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HE_numerator          calojetpT_HE_denominator",
        "effic_calojetpT_HE_variable 'Jet pT turnON (HE);            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HE_numerator calojetpT_variable_HE_denominator",
        "effic_calojetPhi_HE       'Jet efficiency vs #phi (HE); Calo Jet phi [rad]; efficiency' calojetPhi_HE_numerator       calojetPhi_HE_denominator",
        "effic_calojetEta_HE       'Jet efficiency vs #eta (HE); Calo Jet phi [rad]; efficiency' calojetEta_HE_numerator       calojetEta_HE_denominator",

        "effic_calojetpT_HE_p          'Jet pT turnON (HEP);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HE_p_numerator          calojetpT_HE_p_denominator",
        "effic_calojetpT_HE_p_variable 'Jet pT turnON (HEP);            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HE_p_numerator calojetpT_variable_HE_P_denominator",
        "effic_calojetPhi_HE_p       'Jet efficiency vs #phi (HEP); Calo Jet phi [rad]; efficiency' calojetPhi_HE_p_numerator       calojetPhi_HE_p_denominator",
        "effic_calojetEta_HE_p       'Jet efficiency vs #eta (HEP); Calo Jet phi [rad]; efficiency' calojetEta_HE_p_numerator       calojetEta_HE_p_denominator",

        "effic_calojetpT_HE_m          'Jet pT turnON (HEM);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HE_m_numerator          calojetpT_HE_m_denominator",
        "effic_calojetpT_HE_m_variable 'Jet pT turnON (HEM);            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HE_m_numerator calojetpT_variable_HE_m_denominator",
        "effic_calojetPhi_HE_m       'Jet efficiency vs #phi (HEM); Calo Jet phi [rad]; efficiency' calojetPhi_HE_m_numerator       calojetPhi_HE_m_denominator",
        "effic_calojetEta_HE_m       'Jet efficiency vs #eta (HEM); Calo Jet phi [rad]; efficiency' calojetEta_HE_m_numerator       calojetEta_HE_m_denominator",

        "effic_calojetpT_HEP17          'Jet pT turnON (HEP17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEP17_numerator          calojetpT_HEP17_denominator",
        "effic_calojetpT_HEP17_variable 'Jet pT turnON (HEP17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HEP17_numerator calojetpT_variable_HEP17_denominator",
        "effic_calojetPhi_HEP17       'Jet efficiency vs #phi (HEP17); Calo Jet phi [rad]; efficiency' calojetPhi_HEP17_numerator       calojetPhi_HEP17_denominator",
        "effic_calojetEta_HEP17       'Jet efficiency vs #eta (HEP17); Calo Jet phi [rad]; efficiency' calojetEta_HEP17_numerator       calojetEta_HEP17_denominator",

        "effic_calojetpT_HEP18          'Jet pT turnON (HEP18);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEP18_numerator          calojetpT_HEP18_denominator",
        "effic_calojetpT_HEP18_variable 'Jet pT turnON (HEP18);            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HEP18_numerator calojetpT_variable_HEP18_denominator",
        "effic_calojetPhi_HEP18       'Jet efficiency vs #phi (HEP18); Calo Jet phi [rad]; efficiency' calojetPhi_HEP18_numerator       calojetPhi_HEP18_denominator",
        "effic_calojetEta_HEP18       'Jet efficiency vs #eta (HEP18); Calo Jet phi [rad]; efficiency' calojetEta_HEP18_numerator       calojetEta_HEP18_denominator",

        "effic_calojetpT_HEM17          'Jet pT turnON (HEM17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEM17_numerator          calojetpT_HEM17_denominator",
        "effic_calojetpT_HEM17_variable 'Jet pT turnON (HEM17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HEM17_numerator calojetpT_variable_HEM17_denominator",
        "effic_calojetPhi_HEM17       'Jet efficiency vs #phi (HEM17); Calo Jet phi [rad]; efficiency' calojetPhi_HEM17_numerator       calojetPhi_HEM17_denominator",
        "effic_calojetEta_HEM17       'Jet efficiency vs #eta (HEM17); Calo Jet phi [rad]; efficiency' calojetEta_HEM17_numerator       calojetEta_HEM17_denominator",

        "effic_calojetpT_HF          'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_HF_numerator          calojetpT_HF_denominator",
        "effic_calojetpT_HF_variable 'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_variable_HF_numerator calojetpT_variable_HF_denominator",
        "effic_calojetPhi_HF       'Jet efficiency vs #phi; Calo Jet phi [rad]; efficiency' calojetPhi_HF_numerator       calojetPhi_HF_denominator",
        "effic_calojetEta_HF       'Jet efficiency vs #eta; Calo Jet phi [rad]; efficiency' calojetEta_HF_numerator       calojetEta_HF_denominator",

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

pfmetEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JetMET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_pfmetpT          'MET pT turnON;            PFMET(pT) [GeV]; efficiency'     pfmetpT_numerator          pfmetpT_denominator",
        "effic_pfmetpT_variable 'MET pT turnON;            PFMET(pT) [GeV]; efficiency'     pfmetpT_variable_numerator pfmetpT_variable_denominator",
        "effic_pfmetPhi       'MET efficiency vs #phi; PF MET phi [rad]; efficiency' pfmetPhi_numerator       pfmetPhi_denominator",
        "effic_pfmetEta       'MET efficiency vs #eta; PF MET phi [rad]; efficiency' pfmetEta_numerator       pfmetEta_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' pfmetpTVsLS_numerator pfmetpTVsLS_denominator"
    ),
  
)


JetMetPromClient = cms.Sequence(
    pfjetEfficiency
    *calojetEfficiency
    *pfmetEfficiency
)
