import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
pfjetEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_pfpuppijetpT          'Jet pT turnON;            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_numerator          pfpuppijetpT_denominator",
        "effic_pfpuppijetpT_pTThresh 'Jet pT turnON;            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_pTThresh_numerator pfpuppijetpT_pTThresh_denominator",
        "effic_pfpuppijetphi       'Jet efficiency vs #phi; PFPuppi Jet #phi [rad]; efficiency' pfpuppijetphi_numerator       pfpuppijetphi_denominator",
        "effic_pfpuppijeteta       'Jet efficiency vs #eta; PFPuppi Jet #eta; efficiency' pfpuppijeteta_numerator       pfpuppijeteta_denominator",
        ## HB
        "effic_pfpuppijetpT_HB          'Jet pT turnON (HB);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_HB_numerator          pfpuppijetpT_HB_denominator",
        "effic_pfpuppijetpT_HB_pTThresh 'Jet pT turnON (HB);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_pTThresh_HB_numerator pfpuppijetpT_pTThresh_HB_denominator",
        "effic_pfpuppijetphi_HB       'Jet efficiency vs #phi (HB); PFPuppi Jet #phi [rad]; efficiency' pfpuppijetphi_HB_numerator       pfpuppijetphi_HB_denominator",
        "effic_pfpuppijeteta_HB       'Jet efficiency vs #eta (HB); PFPuppi Jet #eta; efficiency' pfpuppijeteta_HB_numerator       pfpuppijeteta_HB_denominator",
        ## HE
        "effic_pfpuppijetpT_HE          'Jet pT turnON (HE);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_HE_numerator          pfpuppijetpT_HE_denominator",
        "effic_pfpuppijetpT_HE_pTThresh 'Jet pT turnON (HE);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_pTThresh_HE_numerator pfpuppijetpT_pTThresh_HE_denominator",
        "effic_pfpuppijetphi_HE       'Jet efficiency vs #phi (HE); PFPuppi Jet #phi [rad]; efficiency' pfpuppijetphi_HE_numerator       pfpuppijetphi_HE_denominator",
        "effic_pfpuppijeteta_HE       'Jet efficiency vs #eta (HE); PFPuppi Jet #eta; efficiency' pfpuppijeteta_HE_numerator       pfpuppijeteta_HE_denominator",
         ## HE_p
        "effic_pfpuppijetpT_HE_p          'Jet pT turnON (HEP);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_HE_p_numerator          pfpuppijetpT_HE_p_denominator",
        "effic_pfpuppijetpT_HE_p_pTThresh 'Jet pT turnON (HEP);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_pTThresh_HE_p_numerator pfpuppijetpT_pTThresh_HE_p_denominator",
        ## HE_m
        "effic_pfpuppijetpT_HE_m          'Jet pT turnON (HEM);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_HE_m_numerator          pfpuppijetpT_HE_m_denominator",
        "effic_pfpuppijetpT_HE_m_pTThresh 'Jet pT turnON (HEM);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_pTThresh_HE_m_numerator pfpuppijetpT_pTThresh_HE_m_denominator", 
        ## HF
        "effic_pfpuppijetpT_HF          'Jet pT turnON (HF);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_HF_numerator          pfpuppijetpT_HF_denominator",
        "effic_pfpuppijetpT_HF_pTThresh 'Jet pT turnON (HF);            PFPuppi Jet(pT) [GeV]; efficiency'     pfpuppijetpT_pTThresh_HF_numerator pfpuppijetpT_pTThresh_HF_denominator",
        "effic_pfpuppijetphi_HF       'Jet efficiency vs #phi (HF); PFPuppi Jet #phi [rad]; efficiency' pfpuppijetphi_HF_numerator       pfpuppijetphi_HF_denominator",
        "effic_pfpuppijeteta_HF       'Jet efficiency vs #eta (HF); PFPuppi Jet #eta; efficiency' pfpuppijeteta_HF_numerator       pfpuppijeteta_HF_denominator",
        ## 2D Eff
        "effic_pfpuppijetEtaVsPhi       'Jet efficiency vs #eta and #phi; PFPuppi Jet #eta; #phi' pfpuppijetEtaVsPhi_numerator       pfpuppijetEtaVsPhi_denominator",
        #"effic_pfpuppijetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB); PFPuppi Jet #eta; #phi' pfpuppijetEtaVsPhi_HB_numerator       pfpuppijetEtaVsPhi_HB_denominator",
        #"effic_pfpuppijetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE); PFPuppi Jet #eta; #phi' pfpuppijetEtaVsPhi_HE_numerator       pfpuppijetEtaVsPhi_HE_denominator",
        #"effic_pfpuppijetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF); PFPuppi Jet #eta; #phi' pfpuppijetEtaVsPhi_HF_numerator       pfpuppijetEtaVsPhi_HF_denominator",
         
        "effic_pfpuppijetEtaVspT        'Jet efficiency #eta vs Pt;     PFPuppi Jet #eta; Pt' pfpuppijetEtaVspT_numerator          pfpuppijetEtaVspT_denominator",
        #"effic_pfpuppijetEtaVspT_HB     'Jet efficiency #eta vs Pt(HB); PFPuppi Jet #eta; Pt' pfpuppijetEtaVspT_HB_numerator       pfpuppijetEtaVspT_HB_denominator",
        #"effic_pfpuppijetEtaVspT_HE     'Jet efficiency #eta vs Pt(HE); PFPuppi Jet #eta; Pt' pfpuppijetEtaVspT_HE_numerator       pfpuppijetEtaVspT_HE_denominator",
        #"effic_pfpuppijetEtaVspT_HF     'Jet efficiency #eta vs Pt(HF); PFPuppi Jet #eta; Pt' pfpuppijetEtaVspT_HF_numerator       pfpuppijetEtaVspT_HF_denominator",

    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_pfpuppijetpT_vs_LS 'JET efficiency vs LS; LS; PF JET efficiency' pfpuppijetpTVsLS_numerator pfpuppijetpTVsLS_denominator",
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

        "effic_calojetpT_HE_m          'Jet pT turnON (HEM);            CaloJet(pT) [GeV]; efficiency'     calojetpT_HE_m_numerator          calojetpT_HE_m_denominator",
        "effic_calojetpT_HE_m_pTThresh 'Jet pT turnON (HEM);            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HE_m_numerator calojetpT_pTThresh_HE_m_denominator",

        "effic_calojetpT_HF          'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_HF_numerator          calojetpT_HF_denominator",
        "effic_calojetpT_HF_pTThresh 'Jet pT turnON;            CaloJet(pT) [GeV]; efficiency'     calojetpT_pTThresh_HF_numerator calojetpT_pTThresh_HF_denominator",
        "effic_calojetphi_HF       'Jet efficiency vs #phi; Calo Jet #phi [rad]; efficiency' calojetphi_HF_numerator       calojetphi_HF_denominator",
        "effic_calojeteta_HF       'Jet efficiency vs #eta; Calo Jet #eta; efficiency' calojeteta_HF_numerator       calojeteta_HF_denominator",

        ## 2D Eff
        "effic_calojetEtaVsPhi       'Jet efficiency vs #eta and #phi;      Calo Jet #eta; #phi' calojetEtaVsPhi_numerator          calojetEtaVsPhi_denominator",
        #"effic_calojetEtaVsPhi_HB    'Jet efficiency vs #eta and #phi(HB);  Calo Jet #eta; #phi' calojetEtaVsPhi_HB_numerator       calojetEtaVsPhi_HB_denominator",
        #"effic_calojetEtaVsPhi_HE    'Jet efficiency vs #eta and #phi(HE);  Calo Jet #eta; #phi' calojetEtaVsPhi_HE_numerator       calojetEtaVsPhi_HE_denominator",
        #"effic_calojetEtaVsPhi_HF    'Jet efficiency vs #eta and #phi(HF);  Calo Jet #eta; #phi' calojetEtaVsPhi_HF_numerator       calojetEtaVsPhi_HF_denominator",
        
        "effic_calojetEtaVspT        'Jet efficiency #eta vs Pt;        Calo Jet #eta; Pt' calojetEtaVspT_numerator          calojetEtaVspT_denominator",
        #"effic_calojetEtaVspT_HB     'Jet efficiency #eta vs Pt(HB);    Calo Jet #eta; Pt' calojetEtaVspT_HB_numerator       calojetEtaVspT_HB_denominator",
        #"effic_calojetEtaVspT_HE     'Jet efficiency #eta vs Pt(HE);    Calo Jet #eta; Pt' calojetEtaVspT_HE_numerator       calojetEtaVspT_HE_denominator",
        #"effic_calojetEtaVspT_HF     'Jet efficiency #eta vs Pt(HF);    Calo Jet #eta; Pt' calojetEtaVspT_HF_numerator       calojetEtaVspT_HF_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_calojetpT_vs_LS 'JET efficiency vs LS; LS; Calo Jet efficiency' calojetpTVsLS_numerator calojetpTVsLS_denominator",
    ),
  
)

GammaJetsEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/ZGammaPlusJets*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_HLTJetPt40           'HLT Jet (cut=40GeV) efficiency vs HLT Photon Pt [GeV];             HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over40_numerator           PhotonpT_numerator",
        "effic_HLTJetPt40_HB        'HLT Jet (cut=40GeV) efficiency [HB] vs HLT Photon Pt [GeV];        HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over40_HB_numerator        PhotonpT_HB_numerator",
        "effic_HLTJetPt40_HEInner   'HLT Jet (cut=40GeV) efficiency [HEInner] vs HLT Photon Pt [GeV];   HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over40_HEInner_numerator   PhotonpT_HEInner_numerator",
        "effic_HLTJetPt40_HEOuter   'HLT Jet (cut=40GeV) efficiency [HEOuter] vs HLT Photon Pt [GeV];   HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over40_HEOuter_numerator   PhotonpT_HEOuter_numerator",

        "effic_HLTJetPt60           'HLT Jet (cut=60GeV) efficiency vs HLT Photon Pt [GeV];             HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over60_numerator           PhotonpT_numerator",
        "effic_HLTJetPt60_HB        'HLT Jet (cut=60GeV) efficiency [HB] vs HLT Photon Pt [GeV];        HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over60_HB_numerator        PhotonpT_HB_numerator",
        "effic_HLTJetPt60_HEInner   'HLT Jet (cut=60GeV) efficiency [HEInner] vs HLT Photon Pt [GeV];   HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over60_HEInner_numerator   PhotonpT_HEInner_numerator",
        "effic_HLTJetPt60_HEOuter   'HLT Jet (cut=60GeV) efficiency [HEOuter] vs HLT Photon Pt [GeV];   HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over60_HEOuter_numerator   PhotonpT_HEOuter_numerator",

        "effic_HLTJetPt80           'HLT Jet (cut=80GeV) efficiency vs HLT Photon Pt [GeV];             HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over80_numerator           PhotonpT_numerator",
        "effic_HLTJetPt80_HB        'HLT Jet (cut=80GeV) efficiency [HB] vs HLT Photon Pt [GeV];        HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over80_HB_numerator        PhotonpT_HB_numerator",
        "effic_HLTJetPt80_HEInner   'HLT Jet (cut=80GeV) efficiency [HEInner] vs HLT Photon Pt [GeV];   HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over80_HEInner_numerator   PhotonpT_HEInner_numerator",
        "effic_HLTJetPt80_HEOuter   'HLT Jet (cut=80GeV) efficiency [HEOuter] vs HLT Photon Pt [GeV];   HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over80_HEOuter_numerator   PhotonpT_HEOuter_numerator",

        "effic_HLTJetPt110          'HLT Jet (cut=110GeV) efficiency vs HLT Photon Pt [GeV];            HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over110_numerator          PhotonpT_numerator",
        "effic_HLTJetPt110_HB       'HLT Jet (cut=110GeV) efficiency [HB] vs HLT Photon Pt [GeV];       HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over110_HB_numerator       PhotonpT_HB_numerator",
        "effic_HLTJetPt110_HEInner  'HLT Jet (cut=110GeV) efficiency [HEInner] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over110_HEInner_numerator  PhotonpT_HEInner_numerator",
        "effic_HLTJetPt110_HEOuter  'HLT Jet (cut=110GeV) efficiency [HEOuter] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over110_HEOuter_numerator  PhotonpT_HEOuter_numerator",
        
        "effic_HLTJetPt140          'HLT Jet (cut=140GeV) efficiency vs HLT Photon Pt [GeV];            HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over140_numerator          PhotonpT_numerator",
        "effic_HLTJetPt140_HB       'HLT Jet (cut=140GeV) efficiency [HB] vs HLT Photon Pt [GeV];       HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over140_HB_numerator       PhotonpT_HB_numerator",
        "effic_HLTJetPt140_HEInner  'HLT Jet (cut=140GeV) efficiency [HEInner] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over140_HEInner_numerator  PhotonpT_HEInner_numerator",
        "effic_HLTJetPt140_HEOuter  'HLT Jet (cut=140GeV) efficiency [HEOuter] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over140_HEOuter_numerator  PhotonpT_HEOuter_numerator",

        "effic_HLTJetPt200          'HLT Jet (cut=200GeV) efficiency vs HLT Photon Pt [GeV];            HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over200_numerator          PhotonpT_numerator",
        "effic_HLTJetPt200_HB       'HLT Jet (cut=200GeV) efficiency [HB] vs HLT Photon Pt [GeV];       HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over200_HB_numerator       PhotonpT_HB_numerator",
        "effic_HLTJetPt200_HEInner  'HLT Jet (cut=200GeV) efficiency [HEInner] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over200_HEInner_numerator  PhotonpT_HEInner_numerator",
        "effic_HLTJetPt200_HEOuter  'HLT Jet (cut=200GeV) efficiency [HEOuter] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over200_HEOuter_numerator  PhotonpT_HEOuter_numerator",

        "effic_HLTJetPt320          'HLT Jet (cut=320GeV) efficiency vs HLT Photon Pt [GeV];            HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over320_numerator          PhotonpT_numerator",
        "effic_HLTJetPt320_HB       'HLT Jet (cut=320GeV) efficiency [HB] vs HLT Photon Pt [GeV];       HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over320_HB_numerator       PhotonpT_HB_numerator",
        "effic_HLTJetPt320_HEInner  'HLT Jet (cut=320GeV) efficiency [HEInner] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over320_HEInner_numerator  PhotonpT_HEInner_numerator",
        "effic_HLTJetPt320_HEOuter  'HLT Jet (cut=320GeV) efficiency [HEOuter] vs HLT Photon Pt [GeV];  HLT Photon (pT) [GeV]; efficiency'  PhotonpT_for_JetpT_over320_HEOuter_numerator  PhotonpT_HEOuter_numerator",
        ),
    )
ZJetsEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/ZGammaPlusJets*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_HLTJetPt40           'HLT Jet (cut=40GeV) efficiency vs HLT Z Pt [GeV];             HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over40_numerator           ZpT_numerator",
        "effic_HLTJetPt40_HB        'HLT Jet (cut=40GeV) efficiency [HB] vs HLT Z Pt [GeV];        HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over40_HB_numerator        ZpT_HB_numerator",
        "effic_HLTJetPt40_HEInner   'HLT Jet (cut=40GeV) efficiency [HEInner] vs HLT Z Pt [GeV];   HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over40_HEInner_numerator   ZpT_HEInner_numerator",
        "effic_HLTJetPt40_HEOuter   'HLT Jet (cut=40GeV) efficiency [HEOuter] vs HLT Z Pt [GeV];   HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over40_HEOuter_numerator   ZpT_HEOuter_numerator",

        "effic_HLTJetPt60           'HLT Jet (cut=60GeV) efficiency vs HLT Z Pt [GeV];             HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over60_numerator           ZpT_numerator",
        "effic_HLTJetPt60_HB        'HLT Jet (cut=60GeV) efficiency [HB] vs HLT Z Pt [GeV];        HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over60_HB_numerator        ZpT_HB_numerator",
        "effic_HLTJetPt60_HEInner   'HLT Jet (cut=60GeV) efficiency [HEInner] vs HLT Z Pt [GeV];   HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over60_HEInner_numerator   ZpT_HEInner_numerator",
        "effic_HLTJetPt60_HEOuter   'HLT Jet (cut=60GeV) efficiency [HEOuter] vs HLT Z Pt [GeV];   HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over60_HEOuter_numerator   ZpT_HEOuter_numerator",

        "effic_HLTJetPt80           'HLT Jet (cut=80GeV) efficiency vs HLT Z Pt [GeV];             HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over80_numerator           ZpT_numerator",
        "effic_HLTJetPt80_HB        'HLT Jet (cut=80GeV) efficiency [HB] vs HLT Z Pt [GeV];        HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over80_HB_numerator        ZpT_HB_numerator",
        "effic_HLTJetPt80_HEInner   'HLT Jet (cut=80GeV) efficiency [HEInner] vs HLT Z Pt [GeV];   HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over80_HEInner_numerator   ZpT_HEInner_numerator",
        "effic_HLTJetPt80_HEOuter   'HLT Jet (cut=80GeV) efficiency [HEOuter] vs HLT Z Pt [GeV];   HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over80_HEOuter_numerator   ZpT_HEOuter_numerator",
        
        "effic_HLTJetPt110          'HLT Jet (cut=110GeV) efficiency vs HLT Z Pt [GeV];            HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over110_numerator          ZpT_numerator",
        "effic_HLTJetPt110_HB       'HLT Jet (cut=110GeV) efficiency [HB] vs HLT Z Pt [GeV];       HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over110_HB_numerator       ZpT_HB_numerator",
        "effic_HLTJetPt110_HEInner  'HLT Jet (cut=110GeV) efficiency [HEInner] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over110_HEInner_numerator  ZpT_HEInner_numerator",
        "effic_HLTJetPt110_HEOuter  'HLT Jet (cut=110GeV) efficiency [HEOuter] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over110_HEOuter_numerator  ZpT_HEOuter_numerator",
        
        "effic_HLTJetPt140          'HLT Jet (cut=140GeV) efficiency vs HLT Z Pt [GeV];            HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over140_numerator          ZpT_numerator",
        "effic_HLTJetPt140_HB       'HLT Jet (cut=140GeV) efficiency [HB] vs HLT Z Pt [GeV];       HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over140_HB_numerator       ZpT_HB_numerator",
        "effic_HLTJetPt140_HEInner  'HLT Jet (cut=140GeV) efficiency [HEInner] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over140_HEInner_numerator  ZpT_HEInner_numerator",
        "effic_HLTJetPt140_HEOuter  'HLT Jet (cut=140GeV) efficiency [HEOuter] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over140_HEOuter_numerator  ZpT_HEOuter_numerator",

        "effic_HLTJetPt200          'HLT Jet (cut=200GeV) efficiency vs HLT Z Pt [GeV];            HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over200_numerator          ZpT_numerator",
        "effic_HLTJetPt200_HB       'HLT Jet (cut=200GeV) efficiency [HB] vs HLT Z Pt [GeV];       HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over200_HB_numerator       ZpT_HB_numerator",
        "effic_HLTJetPt200_HEInner  'HLT Jet (cut=200GeV) efficiency [HEInner] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over200_HEInner_numerator  ZpT_HEInner_numerator",
        "effic_HLTJetPt200_HEOuter  'HLT Jet (cut=200GeV) efficiency [HEOuter] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over200_HEOuter_numerator  ZpT_HEOuter_numerator",

        "effic_HLTJetPt320          'HLT Jet (cut=320GeV) efficiency vs HLT Z Pt [GeV];            HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over320_numerator          ZpT_numerator",
        "effic_HLTJetPt320_HB       'HLT Jet (cut=320GeV) efficiency [HB] vs HLT Z Pt [GeV];       HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over320_HB_numerator       ZpT_HB_numerator",
        "effic_HLTJetPt320_HEInner  'HLT Jet (cut=320GeV) efficiency [HEInner] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over320_HEInner_numerator  ZpT_HEInner_numerator",
        "effic_HLTJetPt320_HEOuter  'HLT Jet (cut=320GeV) efficiency [HEOuter] vs HLT Z Pt [GeV];  HLT Z (pT) [GeV]; efficiency'  ZpT_for_JetpT_over320_HEOuter_numerator  ZpT_HEOuter_numerator",
        ),
    )

#-----------------------------

JetMetPromClient = cms.Sequence(
    pfjetEfficiency
    *calojetEfficiency
    *GammaJetsEfficiency
    *ZJetsEfficiency
)
