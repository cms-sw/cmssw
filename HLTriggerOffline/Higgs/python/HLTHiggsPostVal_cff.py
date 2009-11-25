import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Higgs.PostProcessor_cfi import *

hltHiggsPostHWW = HLTHiggsPostProcessor.clone()

hltHiggsPostHWW.subDirs        = ['HLT/Higgs/HWW']  
hltHiggsPostHWW.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    
    "EffVsPt_HLT_Mu3                                'HLT_Mu3                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu3                                Muon1Pt",
    "EffVsEta_HLT_Mu3                               'HLT_Mu3                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu3                               Muon1Eta", 
    
    "EffVsPt_HLT_Mu9                                'HLT_Mu9                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu9                                Muon1Pt",
    "EffVsEta_HLT_Mu9                               'HLT_Mu9                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu9                               Muon1Eta",
   
    "EffVsPt_HLT_Mu15                                'HLT_Mu15                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu15                                Muon1Pt",
    "EffVsEta_HLT_Mu15                               'HLT_Mu15                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu15                               Muon1Eta",
   
    "EffVsPt_HLT_Ele10_LW_L1R                       'HLT_Ele10_LW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele10_LW_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele10_LW_L1R                      'HLT_Ele10_LW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele10_LW_L1R                  Electron1Eta",
   
   
    "EffVsPt_HLT_Ele15_SW_L1R                       'HLT_Ele15_SW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele15_SW_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele15_SW_L1R                      'HLT_Ele15_SW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele15_SW_L1R                  Electron1Eta",
   
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R                       'HLT_Ele15_SW_LooseTrackIso_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele15_SW_LooseTrackIso_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R                      'HLT_Ele15_SW_LooseTrackIso_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele15_SW_LooseTrackIso_L1R                  Electron1Eta",
   
     ########
     
    "EffVsPt_HLT_Mu3_EM                                'HLT_Mu3                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu3                                Muon1Pt_EM",
    "EffVsEta_HLT_Mu3_EM                               'HLT_Mu3                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu3                               Muon1Eta_EM", 
    
    "EffVsPt_HLT_Mu9_EM                                'HLT_Mu9                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu9                                Muon1Pt_EM",
    "EffVsEta_HLT_Mu9_EM                               'HLT_Mu9                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu9                               Muon1Eta_EM",
   
    "EffVsPt_HLT_Mu15_EM                                'HLT_Mu15                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu15                                Muon1Pt_EM",
    "EffVsEta_HLT_Mu15_EM                               'HLT_Mu15                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu15                               Muon1Eta_EM",
   
    "EffVsPt_HLT_Ele10_LW_L1R_EM                       'HLT_Ele10_LW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele10_LW_L1R                   Electron1Pt_EM",
    "EffVsEta_HLT_Ele10_LW_L1R_EM                      'HLT_Ele10_LW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele10_LW_L1R                  Electron1Eta_EM",
   
   
    "EffVsPt_HLT_Ele15_SW_L1R_EM                       'HLT_Ele15_SW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele15_SW_L1R                   Electron1Pt_EM",
    "EffVsEta_HLT_Ele15_SW_L1R_EM                      'HLT_Ele15_SW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele15_SW_L1R                  Electron1Eta_EM",
   
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R_EM                       'HLT_Ele15_SW_LooseTrackIso_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele15_SW_LooseTrackIso_L1R                   Electron1Pt_EM",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R_EM                      'HLT_Ele15_SW_LooseTrackIso_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele15_SW_LooseTrackIso_L1R                  Electron1Eta_EM",
   
    "Efficiencies_MuonTriggers                 'HLT efficiency wrt reco selection        ;  ;       Trigger Efficiency'      muHLT   MuonEvents",
    
    "Efficiencies_ElectronTriggers             'HLT efficiency wrt reco selection       ; ;         Trigger Efficiency'  elHLT   ElectronEvents",
   
    "TriggerEfficiencies_EmuChannel            'HLT efficiency wrt reco selection       ; ;         Trigger Efficiency'  emuHLT   EmuEvents",
    

   
   
    )
    
hltHiggsPostHgg = HLTHiggsPostProcessor.clone()

hltHiggsPostHgg.subDirs        = ['HLT/Higgs/Hgg']  
hltHiggsPostHgg.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    
    "EffVsPt_HLT_Photon15_L1R                                'HLT_Photon15_L1R                           ; p_{T #gamma};        Trigger_Efficiency'  Photon1Pt_HLT_Photon15_L1R                                Photon1Pt",
    "EffVsEta_HLT_Photon15_L1R                               'HLT_Photon15_L1R                           ; #eta_{#gamma};       Trigger_Efficiency'  Photon1Eta_HLT_Photon15_L1R                               Photon1Eta",
   
    "EffVsPt_HLT_Photon15_TrackIso_L1R                                'HLT_Photon15_TrackIso_L1R                           ; p_{T #gamma};        Trigger_Efficiency'  Photon1Pt_HLT_Photon15_TrackIso_L1R                                Photon1Pt",
    "EffVsEta_HLT_Photon15_TrackIso_L1R                               'HLT_Photon15_TrackIso_L1R                           ; #eta_{#gamma};       Trigger_Efficiency'  Photon1Eta_HLT_Photon15_TrackIso_L1R                               Photon1Eta",
    
    "EffVsPt_HLT_DoublePhoton10_L1R                                'HLT_DoublePhoton10_L1R                           ; p_{T #gamma};        Trigger_Efficiency'  Photon1Pt_HLT_DoublePhoton10_L1R                                Photon1Pt",
    "EffVsEta_HLT_DoublePhoton10_L1R                               'HLT_DoublePhoton10_L1R                           ; #eta_{#gamma};       Trigger_Efficiency'  Photon1Eta_HLT_DoublePhoton10_L1R                               Photon1Eta",
   
    "Efficiencies_PhotonTriggers                      'HLT efficiency wrt reco selection       ;  ;              Trigger Efficiency'  phHLT   PhotonEvents",
    
    )
    
hltHiggsPostHZZ = HLTHiggsPostProcessor.clone()

hltHiggsPostHZZ.subDirs        = ['HLT/Higgs/HZZ']  
hltHiggsPostHZZ.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffVsPt_HLT_Mu3                                'HLT_Mu3                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu3                                Muon1Pt",
    "EffVsEta_HLT_Mu3                               'HLT_Mu3                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu3                               Muon1Eta", 
    
    "EffVsPt_HLT_Mu9                                'HLT_Mu9                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu9                                Muon1Pt",
    "EffVsEta_HLT_Mu9                               'HLT_Mu9                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu9                               Muon1Eta",
   
    "EffVsPt_HLT_Mu15                                'HLT_Mu15                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu15                                Muon1Pt",
    "EffVsEta_HLT_Mu15                               'HLT_Mu15                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu15                               Muon1Eta",
   
    "EffVsPt_HLT_Ele10_LW_L1R                       'HLT_Ele10_LW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele10_LW_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele10_LW_L1R                      'HLT_Ele10_LW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele10_LW_L1R                  Electron1Eta",
   
   
    "EffVsPt_HLT_Ele15_SW_L1R                       'HLT_Ele15_SW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele15_SW_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele15_SW_L1R                      'HLT_Ele15_SW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele15_SW_L1R                  Electron1Eta",
   
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R                       'HLT_Ele15_SW_LooseTrackIso_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele15_SW_LooseTrackIso_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R                      'HLT_Ele15_SW_LooseTrackIso_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele15_SW_LooseTrackIso_L1R                  Electron1Eta",
   
    ########
     
    "EffVsPt_HLT_Mu3_EM                                'HLT_Mu3                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu3                                Muon1Pt_EM",
    "EffVsEta_HLT_Mu3_EM                               'HLT_Mu3                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu3                               Muon1Eta_EM", 
    
    "EffVsPt_HLT_Mu9_EM                                'HLT_Mu9                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu9                                Muon1Pt_EM",
    "EffVsEta_HLT_Mu9_EM                               'HLT_Mu9                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu9                               Muon1Eta_EM",
   
    "EffVsPt_HLT_Mu15_EM                                'HLT_Mu15                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu15                                Muon1Pt_EM",
    "EffVsEta_HLT_Mu15_EM                               'HLT_Mu15                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu15                               Muon1Eta_EM",
   
    "EffVsPt_HLT_Ele10_LW_L1R_EM                       'HLT_Ele10_LW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele10_LW_L1R                   Electron1Pt_EM",
    "EffVsEta_HLT_Ele10_LW_L1R_EM                      'HLT_Ele10_LW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele10_LW_L1R                  Electron1Eta_EM",
   
   
    "EffVsPt_HLT_Ele15_SW_L1R_EM                       'HLT_Ele15_SW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele15_SW_L1R                   Electron1Pt_EM",
    "EffVsEta_HLT_Ele15_SW_L1R_EM                      'HLT_Ele15_SW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele15_SW_L1R                  Electron1Eta_EM",
   
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R_EM                       'HLT_Ele15_SW_LooseTrackIso_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele15_SW_LooseTrackIso_L1R                   Electron1Pt_EM",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R_EM                      'HLT_Ele15_SW_LooseTrackIso_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele15_SW_LooseTrackIso_L1R                  Electron1Eta_EM",
   
   ######
    "Efficiencies_MuonTriggers                 'HLT efficiency wrt reco selection        ;  ;       Trigger Efficiency'      muHLT   MuonEvents",
    
    "Efficiencies_ElectronTriggers             'HLT efficiency wrt reco selection       ; ;         Trigger Efficiency'  elHLT   ElectronEvents",
   
    "TriggerEfficiencies_EmuChannel            'HLT efficiency wrt reco selection       ; ;         Trigger Efficiency'  emuHLT   EmuEvents",
    
    
    )
    
hltHiggsPostH2tau = HLTHiggsPostProcessor.clone()

hltHiggsPostH2tau.subDirs        = ['HLT/Higgs/H2tau']  
hltHiggsPostH2tau.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    
    "EffVsPt_HLT_Mu3                                'HLT_Mu3                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu3                                Muon1Pt",
    "EffVsEta_HLT_Mu3                               'HLT_Mu3                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu3                               Muon1Eta", 
    
    "EffVsPt_HLT_Mu9                                'HLT_Mu9                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu9                                Muon1Pt",
    "EffVsEta_HLT_Mu9                               'HLT_Mu9                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu9                               Muon1Eta",
   
    "EffVsPt_HLT_Mu15                                'HLT_Mu15                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu15                                Muon1Pt",
    "EffVsEta_HLT_Mu15                               'HLT_Mu15                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu15                               Muon1Eta",
   
    "EffVsPt_HLT_Ele10_LW_L1R                       'HLT_Ele10_LW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele10_LW_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele10_LW_L1R                      'HLT_Ele10_LW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele10_LW_L1R                  Electron1Eta",
   
   
    "EffVsPt_HLT_Ele15_SW_L1R                       'HLT_Ele15_SW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele15_SW_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele15_SW_L1R                      'HLT_Ele15_SW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele15_SW_L1R                  Electron1Eta",
   
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R                       'HLT_Ele15_SW_LooseTrackIso_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele15_SW_LooseTrackIso_L1R                   Electron1Pt",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R                      'HLT_Ele15_SW_LooseTrackIso_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele15_SW_LooseTrackIso_L1R                  Electron1Eta",
     
     #####
    
    "Efficiencies_MuonTriggers                 'HLT efficiency wrt MC selection        ;  ;       Trigger Efficiency'      muHLT   MuonEvents",
    
    "Efficiencies_ElectronTriggers             'HLT efficiency wrt MC selection       ; ;         Trigger Efficiency'  elHLT   ElectronEvents",
   
     )
     
     
hltHiggsPostHtaunu = HLTHiggsPostProcessor.clone()

hltHiggsPostHtaunu.subDirs        = ['HLT/Higgs/Htaunu']  
hltHiggsPostHtaunu.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    

    "Efficiencies_TauTriggers                 'HLT efficiency wrt MC selection        ;  ;       Trigger Efficiency'      tauHLT   tauEvents",
    
   
    )
    
HLTHiggsPostVal = cms.Sequence(
    hltHiggsPostHgg   *
    hltHiggsPostHWW *
    hltHiggsPostHZZ *
    hltHiggsPostH2tau *
    hltHiggsPostHtaunu
)

