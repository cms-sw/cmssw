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

        "effic_pfjetpT_HE          'Jet pT turnON (HE);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_numerator          pfjetpT_HE_denominator",
        "effic_pfjetpT_HE_pTThresh 'Jet pT turnON (HE);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HE_numerator pfjetpT_pTThresh_HE_denominator",
        "effic_pfjetphi_HE       'Jet efficiency vs #phi (HE); PF Jet #phi [rad]; efficiency' pfjetphi_HE_numerator       pfjetphi_HE_denominator",
        "effic_pfjeteta_HE       'Jet efficiency vs #eta (HE); PF Jet #eta; efficiency' pfjeteta_HE_numerator       pfjeteta_HE_denominator",

        "effic_pfjetpT_HE_p          'Jet pT turnON (HEP);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_p_numerator          pfjetpT_HE_p_denominator",
        "effic_pfjetpT_HE_p_pTThresh 'Jet pT turnON (HEP);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HE_p_numerator pfjetpT_pTThresh_HE_p_denominator",
        "effic_pfjetphi_HE_p       'Jet efficiency vs #phi (HEP); PF Jet #phi [rad]; efficiency' pfjetphi_HE_p_numerator       pfjetphi_HE_p_denominator",
        "effic_pfjeteta_HE_p       'Jet efficiency vs #eta (HEP); PF Jet #eta; efficiency' pfjeteta_HE_p_numerator       pfjeteta_HE_p_denominator",

        "effic_pfjetpT_HE_m          'Jet pT turnON (HEM);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HE_m_numerator          pfjetpT_HE_m_denominator",
        "effic_pfjetpT_HE_m_pTThresh 'Jet pT turnON (HEM);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HE_m_numerator pfjetpT_pTThresh_HE_m_denominator",
        "effic_pfjetphi_HE_m       'Jet efficiency vs #phi (HEM); PF Jet #phi [rad]; efficiency' pfjetphi_HE_m_numerator       pfjetphi_HE_m_denominator",
        "effic_pfjeteta_HE_m       'Jet efficiency vs #eta (HEM); PF Jet #eta; efficiency' pfjeteta_HE_m_numerator       pfjeteta_HE_m_denominator",

        "effic_pfjetpT_HEP17          'Jet pT turnON (HEP17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HEP17_numerator          pfjetpT_HEP17_denominator",
        "effic_pfjetpT_HEP17_pTThresh 'Jet pT turnON (HEP17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HEP17_numerator pfjetpT_pTThresh_HEP17_denominator",
        "effic_pfjetphi_HEP17       'Jet efficiency vs #phi (HEP17); PF Jet #phi [rad]; efficiency' pfjetphi_HEP17_numerator       pfjetphi_HEP17_denominator",
#        "effic_pfjeteta_HEP17       'Jet efficiency vs #eta (HEP17); PF Jet #eta; efficiency' pfjeteta_HEP17_numerator       pfjeteta_HEP17_denominator",
        "effic_pfjetabseta_HEP17       'Jet efficiency vs |#eta| (HEP17); PF Jet |#eta|; efficiency' pfjetabseta_HEP17_numerator       pfjetabseta_HEP17_denominator",

        "effic_pfjetpT_HEM17          'Jet pT turnON (HEM17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HEM17_numerator          pfjetpT_HEM17_denominator",
        "effic_pfjetpT_HEM17_pTThresh 'Jet pT turnON (HEM17);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HEM17_numerator pfjetpT_pTThresh_HEM17_denominator",
        "effic_pfjetphi_HEM17       'Jet efficiency vs #phi (HEM17); PF Jet #phi [rad]; efficiency' pfjetphi_HEM17_numerator       pfjetphi_HEM17_denominator",
#        "effic_pfjeteta_HEM17       'Jet efficiency vs #eta (HEM17); PF Jet #eta; efficiency' pfjeteta_HEM17_numerator       pfjeteta_HEM17_denominator",
        "effic_pfjetabseta_HEM17       'Jet efficiency vs |#eta| (HEM17); PF Jet |#eta|; efficiency' pfjetabseta_HEM17_numerator       pfjetabseta_HEM17_denominator",

        "effic_pfjetpT_HEP18          'Jet pT turnON (HEP18);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HEP18_numerator          pfjetpT_HEP18_denominator",
        "effic_pfjetpT_HEP18_pTThresh 'Jet pT turnON (HEP18);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HEP18_numerator pfjetpT_pTThresh_HEP18_denominator",
#        "effic_pfjetphi_HEP18       'Jet efficiency vs #phi (HEP18); PF Jet #phi [rad]; efficiency' pfjetphi_HEP18_numerator       pfjetphi_HEP18_denominator",
#        "effic_pfjeteta_HEP18       'Jet efficiency vs #eta (HEP18); PF Jet #eta; efficiency' pfjeteta_HEP18_numerator       pfjeteta_HEP18_denominator",

        "effic_pfjetpT_HF          'Jet pT turnON (HF);            PFJet(pT) [GeV]; efficiency'     pfjetpT_HF_numerator          pfjetpT_HF_denominator",
        "effic_pfjetpT_HF_pTThresh 'Jet pT turnON (HF);            PFJet(pT) [GeV]; efficiency'     pfjetpT_pTThresh_HF_numerator pfjetpT_pTThresh_HF_denominator",
        "effic_pfjetphi_HF       'Jet efficiency vs #phi (HF); PF Jet #phi [rad]; efficiency' pfjetphi_HF_numerator       pfjetphi_HF_denominator",
        "effic_pfjeteta_HF       'Jet efficiency vs #eta (HF); PF Jet #eta; efficiency' pfjeteta_HF_numerator       pfjeteta_HF_denominator",
        ## 2D Eff
        "effic_pfjetEtaVsPhi       'Jet efficiency vs #eta and #phi; PF Jet #eta; #phi' pfjetEtaVsPhi_numerator       pfjetEtaVsPhi_denominator",
        "effic_pfjetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB); PF Jet #eta; #phi' pfjetEtaVsPhi_HB_numerator       pfjetEtaVsPhi_HB_denominator",
        "effic_pfjetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_numerator       pfjetEtaVsPhi_HE_denominator",
        "effic_pfjetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF); PF Jet #eta; #phi' pfjetEtaVsPhi_HF_numerator       pfjetEtaVsPhi_HF_denominator",
#        "effic_pfjetEtaVsPhi_HEM17 'Jet efficiency vs #eta and #phi(HEM17); PF Jet #eta; #phi' pfjetEtaVsPhi_HEM17_numerator       pfjetEtaVsPhi_HEM17_denominator",
#        "effic_pfjetEtaVsPhi_HEP17 'Jet efficiency vs #eta and #phi(HEP17); PF Jet #eta; #phi' pfjetEtaVsPhi_HEP17_numerator       pfjetEtaVsPhi_HEP17_denominator",
#        "effic_pfjetEtaVsPhi_HEP18 'Jet efficiency vs #eta and #phi(HEP18); PF Jet #eta; #phi' pfjetEtaVsPhi_HEP18_numerator       pfjetEtaVsPhi_HEP18_denominator",
        "effic_pfjetEtaVsPhi_HE_p 'Jet efficiency vs #eta and #phi(HE_p); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_p_numerator       pfjetEtaVsPhi_HE_p_denominator",
        "effic_pfjetEtaVsPhi_HE_m 'Jet efficiency vs #eta and #phi(HE_m); PF Jet #eta; #phi' pfjetEtaVsPhi_HE_m_numerator       pfjetEtaVsPhi_HE_m_denominator",
#        "effic_pfjetAbsEtaVsPhi_HEM17 'Jet efficiency vs |#eta| and #phi(HEM17); PF Jet |#eta|; #phi' pfjetAbsEtaVsPhi_HEM17_numerator       pfjetAbsEtaVsPhi_HEM17_denominator",
#        "effic_pfjetAbsEtaVsPhi_HEP17 'Jet efficiency vs |#eta| and #phi(HEP17); PF Jet |#eta|; #phi' pfjetAbsEtaVsPhi_HEP17_numerator       pfjetAbsEtaVsPhi_HEP17_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_pfjetpT_vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_numerator pfjetpTVsLS_denominator",
#        "effic_pfjetpT_HBvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HB_numerator pfjetpTVsLS_HB_denominator",
#        "effic_pfjetpT_HEvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HE_numerator pfjetpTVsLS_HE_denominator",
#        "effic_pfjetpT_HFvs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HF_numerator pfjetpTVsLS_HF_denominator",
#        "effic_pfjetpT_HEP17vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HEP17_numerator pfjetpTVsLS_HEP17_denominator",
#        "effic_pfjetpT_HEM17vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfjetpTVsLS_HEM17_numerator pfjetpTVsLS_HEM17_denominator",
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

        "effic_calojetpT_HEP17          'Jet pT turnON (HEP17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEP17_numerator          calojetpT_HEP17_denominator",
        "effic_calojetpT_HEP17_pTThresh 'Jet pT turnON (HEP17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HEP17_numerator calojetpT_pTThresh_HEP17_denominator",
#        "effic_calojetphi_HEP17       'Jet efficiency vs #phi (HEP17); Calo Jet #phi [rad]; efficiency' calojetphi_HEP17_numerator       calojetphi_HEP17_denominator",
#        "effic_calojeteta_HEP17       'Jet efficiency vs #eta (HEP17); Calo Jet #eta; efficiency' calojeteta_HEP17_numerator       calojeteta_HEP17_denominator",
#        "effic_calojetabseta_HEP17       'Jet efficiency vs |#eta| (HEP17); Calo Jet |#eta|; efficiency' calojetabseta_HEP17_numerator       calojetabseta_HEP17_denominator",

        "effic_calojetpT_HEP18          'Jet pT turnON (HEP18);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEP18_numerator          calojetpT_HEP18_denominator",
        "effic_calojetpT_HEP18_pTThresh 'Jet pT turnON (HEP18);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HEP18_numerator calojetpT_pTThresh_HEP18_denominator",
#        "effic_calojetphi_HEP18       'Jet efficiency vs #phi (HEP18); Calo Jet phi [rad]; efficiency' calojetphi_HEP18_numerator       calojetphi_HEP18_denominator",
#        "effic_calojeteta_HEP18       'Jet efficiency vs #eta (HEP18); Calo Jet #eta; efficiency' calojeteta_HEP18_numerator       calojeteta_HEP18_denominator",

        "effic_calojetpT_HEM17          'Jet pT turnON (HEM17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HEM17_numerator          calojetpT_HEM17_denominator",
        "effic_calojetpT_HEM17_pTThresh 'Jet pT turnON (HEM17);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HEM17_numerator calojetpT_pTThresh_HEM17_denominator",
#        "effic_calojetphi_HEM17       'Jet efficiency vs #phi (HEM17); Calo Jet #phi; efficiency' calojetphi_HEM17_numerator       calojetphi_HEM17_denominator",
#        "effic_calojeteta_HEM17       'Jet efficiency vs #eta (HEM17); Calo Jet #eta; efficiency' calojeteta_HEM17_numerator       calojeteta_HEM17_denominator",
#        "effic_calojetabseta_HEM17       'Jet efficiency vs #eta (HEM17); Calo Jet |#eta|; efficiency' calojetabseta_HEM17_numerator       calojetabseta_HEM17_denominator",

        "effic_calojetpT_HF          'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_HF_numerator          calojetpT_HF_denominator",
        "effic_calojetpT_HF_pTThresh 'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HF_numerator calojetpT_pTThresh_HF_denominator",
        "effic_calojetphi_HF       'Jet efficiency vs #phi; Calo Jet #phi [rad]; efficiency' calojetphi_HF_numerator       calojetphi_HF_denominator",
        "effic_calojeteta_HF       'Jet efficiency vs #eta; Calo Jet #eta; efficiency' calojeteta_HF_numerator       calojeteta_HF_denominator",

        ## 2D Eff
        "effic_calojetEtaVsPhi       'Jet efficiency vs #eta and #phi; PF Jet #eta; #phi' calojetEtaVsPhi_numerator       calojetEtaVsPhi_denominator",
        "effic_calojetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB); PF Jet #eta; #phi' calojetEtaVsPhi_HB_numerator       calojetEtaVsPhi_HB_denominator",
        "effic_calojetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE); PF Jet #eta; #phi' calojetEtaVsPhi_HE_numerator       calojetEtaVsPhi_HE_denominator",
        "effic_calojetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF); PF Jet #eta; #phi' calojetEtaVsPhi_HF_numerator       calojetEtaVsPhi_HF_denominator",
#        "effic_calojetEtaVsPhi_HEM17 'Jet efficiency vs #eta and #phi(HEM17); PF Jet #eta; #phi' calojetEtaVsPhi_HEM17_numerator       calojetEtaVsPhi_HEM17_denominator",
#        "effic_calojetEtaVsPhi_HEP17 'Jet efficiency vs #eta and #phi(HEP17); PF Jet #eta; #phi' calojetEtaVsPhi_HEP17_numerator       calojetEtaVsPhi_HEP17_denominator",
#        "effic_calojetEtaVsPhi_HEP18 'Jet efficiency vs #eta and #phi(HEP18); PF Jet #eta; #phi' calojetEtaVsPhi_HEP18_numerator       calojetEtaVsPhi_HEP18_denominator",
        "effic_calojetEtaVsPhi_HE_p 'Jet efficiency vs #eta and #phi(HE_p); PF Jet #eta; #phi' calojetEtaVsPhi_HE_p_numerator       calojetEtaVsPhi_HE_p_denominator",
        "effic_calojetEtaVsPhi_HE_m 'Jet efficiency vs #eta and #phi(HE_m); PF Jet #eta; #phi' calojetEtaVsPhi_HE_m_numerator       calojetEtaVsPhi_HE_m_denominator",
#        "effic_calojetAbsEtaVsPhi_HEM17 'Jet efficiency vs |#eta| and #phi(HEM17); PF Jet |#eta|; #phi' calojetAbsEtaVsPhi_HEM17_numerator       calojetAbsEtaVsPhi_HEM17_denominator",
#        "effic_calojetAbsEtaVsPhi_HEP17 'Jet efficiency vs |#eta| and #phi(HEP17); PF Jet |#eta|; #phi' calojetAbsEtaVsPhi_HEP17_numerator       calojetAbsEtaVsPhi_HEP17_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_calojetpT_vs_LS 'JET efficiency vs LS; LS; Calo Jet efficiency' calojetpTVsLS_numerator calojetpTVsLS_denominator",
#        "effic_calojetpT_vs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_numerator calojetpTVsLS_denominator",
#        "effic_calojetpT_HBvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HB_numerator calojetpTVsLS_HB_denominator",
#        "effic_calojetpT_HEvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_numerator calojetpTVsLS_HE_denominator",
#        "effic_calojetpT_HFvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HF_numerator calojetpTVsLS_HF_denominator",
#        "effic_calojetpT_HEP17vs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HEP17_numerator calojetpTVsLS_HEP17_denominator",
#        "effic_calojetpT_HEM17vs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HEM17_numerator calojetpTVsLS_HEM17_denominator",
#        "effic_calojetpT_HE_mvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_m_numerator calojetpTVsLS_HE_m_denominator",
#        "effic_calojetpT_HE_pvs_LS 'JET efficiency vs LS; LS; Calo JET efficiency' calojetpTVsLS_HE_p_numerator calojetpTVsLS_HE_p_denominator",
    ),
  
)

pfjetRatio = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "ratio_pfjetpT_HEP17VSHEM17          'HEP17/HEM17 vs pT;            PFJet(pT) [GeV]; Ratio'   effic_pfjetpT_HEP17 effic_pfjetpT_HEM17 simpleratio" ,
        "ratio_pfjetpT_pTTresh_HEP17VSHEM17  'HEP17/HEM17 vs pT;            PFJet(pT) [GeV]; Ratio'   effic_pfjetpT_HEP17_pTThresh effic_pfjetpT_HEM17_pTThresh simpleratio" ,
        "ratio_pfjetphi_HEP17VSHEM17         'HEP17/HEM17 vs #phi;          PFJet #phi [GeV]; Ratio'  effic_pfjetphi_HEP17 effic_pfjetphi_HEM17 simpleratio",
        "ratio_pfjeteta_HEP17VSHEM17         'HEP17/HEM17 vs |#eta|;        PFJet(|#eta|) ; Ratio'    effic_pfjetabseta_HEP17 effic_pfjetabseta_HEM17 simpleratio",

        "ratio_pfjetpT_HEP17VSHEP18          'HEP17/HEP18 vs pT;            PFJet(pT) [GeV]; Ratio'   effic_pfjetpT_HEP17 effic_pfjetpT_HEP18 simpleratio" ,
        "ratio_pfjetpT_pTTresh_HEP17VSHEP18  'HEP17/HEP18 vs pT;            PFJet(pT) [GeV]; Ratio'   effic_pfjetpT_HEP17_pTThresh effic_pfjetpT_HEP18_pTThresh simpleratio" ,
    )
)
calojetRatio = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "ratio_calojetpT_HEP17VSHEM17          'HEP17/HEM17 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17 effic_calojetpT_HEM17 simpleratio" ,
        "ratio_calojetpT_pTTresh_HEP17VSHEM17  'HEP17/HEM17 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17_pTThresh effic_calojetpT_HEM17_pTThresh simpleratio" ,
        "ratio_calojetphi_HEP17VSHEM17         'HEP17/HEM17 vs #phi;          CaloJet #phi [GeV]; Ratio'  effic_calojetphi_HEP17 effic_calojetphi_HEM17 simpleratio",
        "ratio_calojeteta_HEP17VSHEM17         'HEP17/HEM17 vs |#eta|;        CaloJet(|#eta|) ; Ratio'    effic_calojetabseta_HEP17 effic_calojetabseta_HEM17 simpleratio",

        "ratio_calojetpT_HEP17VSHEP18          'HEP17/HEP18 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17 effic_calojetpT_HEP18 simpleratio" ,
        "ratio_calojetpT_pTTresh_HEP17VSHEP18  'HEP17/HEP18 vs pT;            CaloJet(pT) [GeV]; Ratio'   effic_calojetpT_HEP17_pTThresh effic_calojetpT_HEP18_pTThresh simpleratio" ,
    )
)

JetMetPromClient = cms.Sequence(
    pfjetEfficiency
    *calojetEfficiency
    *pfjetRatio
    *calojetRatio
)
