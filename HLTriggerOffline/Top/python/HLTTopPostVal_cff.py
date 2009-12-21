import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Top.PostProcessor_cfi import *

hltTopPostSemimu = HLTTopPostProcessor.clone()

hltTopPostSemimu.subDirs        = ['HLT/Top/Semileptonic_muon']  
hltTopPostSemimu.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffVsPt_HLT_Mu9                                'HLT_Mu9                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_M_HLT_Mu9                                Muon1Pt_M",
    "EffVsEta_HLT_Mu9                               'HLT_Mu9                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_M_HLT_Mu9                               Muon1Eta_M",
    "EffVsPt_HLT_Mu15                               'HLT_Mu15                          ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_M_HLT_Mu15                               Muon1Pt_M",
    "EffVsEta_HLT_Mu15                              'HLT_Mu15                          ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_M_HLT_Mu15                              Muon1Eta_M",
    "EffVsPt_HLT_IsoMu9                             'HLT_IsoMu9                        ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_M_HLT_IsoMu9                             Muon1Pt_M",
    "EffVsEta_HLT_IsoMu9                            'HLT_IsoMu9                        ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_M_HLT_IsoMu9                            Muon1Eta_M",
     
    "Efficiencies_MuonTriggers_gen                  'HLT efficiency wrt mc acceptance      ; ;             Trigger Efficiency'               genmuHLT   genMuonEvents",
    "Efficiencies_MuonTriggers                      'HLT efficiency wrt acc+offline        ; ;             Trigger Efficiency'               muHLT      MuonEvents",
  
    )
    
hltTopPostSemiel = HLTTopPostProcessor.clone()

hltTopPostSemiel.subDirs        = ['HLT/Top/Semileptonic_electron']  
hltTopPostSemiel.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffVsPt_HLT_Ele15_SW_L1R                       'HLT_Ele15_SW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_E_HLT_Ele15_SW_L1R                   Electron1Pt_E",
    "EffVsEta_HLT_Ele15_SW_L1R                      'HLT_Ele15_SW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_E_HLT_Ele15_SW_L1R                  Electron1Eta_E",
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R         'HLT_Ele15_SW_LooseTrackIso_L1R    ; p_{T e};          Trigger_Efficiency'  Electron1Pt_E_HLT_Ele15_SW_LooseTrackIso_L1R     Electron1Pt_E",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R        'HLT_Ele15_SW_LooseTrackIso_L1R    ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_E_HLT_Ele15_SW_LooseTrackIso_L1R    Electron1Eta_E",    
  
    "Efficiencies_Electrontriggers                      'HLT efficiency wrt acc+offline        ; ;              Trigger Efficiency'               elHLT      ElectronEvents",   
    "Efficiencies_Electrontriggers_gen                  'HLT efficiency wrt mc acceptance      ; ;              Trigger Efficiency'               genelHLT   genElectronEvents"
    ####
    )
    
 ##################
 
hltTopPostDimu = HLTTopPostProcessor.clone()

hltTopPostDimu.subDirs        = ['HLT/Top/Dileptonic_muon']  
hltTopPostDimu.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffVsPt_HLT_Mu9_MM                             'HLT_Mu9                           ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_Mu9                               Muon1Pt_MM",
    "EffVsEta_HLT_Mu9_MM                            'HLT_Mu9                           ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_Mu9                              Muon1Eta_MM",
    "EffVsPt_HLT_Mu15_MM                            'HLT_Mu15                          ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_Mu15                              Muon1Pt_MM",
    "EffVsEta_HLT_Mu15_MM                           'HLT_Mu15                          ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_Mu15                             Muon1Eta_MM",
    "EffVsPt_HLT_IsoMu9_MM                          'HLT_IsoMu9                        ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_IsoMu9                            Muon1Pt_MM",
    "EffVsEta_HLT_IsoMu9_MM                         'HLT_IsoMu9                        ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_IsoMu9                           Muon1Eta_MM",
    "EffVsPt_HLT_DoubleMu3_MM                       'HLT_DoubleMu3                     ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_DoubleMu3                         Muon1Pt_MM",
    "EffVsEta_HLT_DoubleMu3_MM                      'HLT_DoubleMu3                     ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_DoubleMu3                        Muon1Eta_MM",    
    
    )
    
hltTopPostDiel = HLTTopPostProcessor.clone()

hltTopPostDiel.subDirs        = ['HLT/Top/Dileptonic_electron']  
hltTopPostDiel.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffVsPt_HLT_Ele15_SW_L1R_EE                    'HLT_Ele15_SW_L1R                  ; p_{T e 1};        Trigger_Efficiency'  Electron1Pt_EE_HLT_Ele15_SW_L1R                  Electron1Pt_EE",
    "EffVsEta_HLT_Ele15_SW_L1R_EE                   'HLT_Ele15_SW_L1R                  ; #eta_{e 1};       Trigger_Efficiency'  Electron1Eta_EE_HLT_Ele15_SW_L1R                 Electron1Eta_EE",
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R_EE      'HLT_Ele15_SW_LooseTrackIso_L1R    ; p_{T e 1};        Trigger_Efficiency'  Electron1Pt_EE_HLT_Ele15_SW_LooseTrackIso_L1R    Electron1Pt_EE",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R_EE     'HLT_Ele15_SW_LooseTrackIso_L1R    ; #eta_{e 1};       Trigger_Efficiency'  Electron1Eta_EE_HLT_Ele15_SW_LooseTrackIso_L1R   Electron1Eta_EE",   
    "EffVsPt_HLT_DoubleEle10_SW_L1R_EE              'HLT_DoubleEle10_SW_L1R            ; p_{T e 1};        Trigger_Efficiency'  Electron1Pt_EE_HLT_DoubleEle10_SW_L1R            Electron1Pt_EE",
    "EffVsEta_HLT_DoubleEle10_SW_L1R_EE             'HLT_DoubleEle10_SW_L1R            ; #eta_{e 1};       Trigger_Efficiency'  Electron1Eta_EE_HLT_DoubleEle10_SW_L1R           Electron1Eta_EE",    

    )
    
hltTopPostEmu = HLTTopPostProcessor.clone()

hltTopPostEmu.subDirs        = ['HLT/Top/Dileptonic_emu']  
hltTopPostEmu.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffVsPt_HLT_Mu9_EM                             'HLT_Mu9                           ; p_{T #mu };       Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu9                               Muon1Pt_EM",
    "EffVsEta_HLT_Mu9_EM                            'HLT_Mu9                           ; #eta_{#mu };      Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu9                              Muon1Eta_EM",
    "EffVsPt_HLT_Mu15_EM                            'HLT_Mu15                          ; p_{T #mu };       Trigger_Efficiency'  Muon1Pt_EM_HLT_Mu15                              Muon1Pt_EM",
    "EffVsEta_HLT_Mu15_EM                           'HLT_Mu15                          ; #eta_{#mu };      Trigger_Efficiency'  Muon1Eta_EM_HLT_Mu15                             Muon1Eta_EM",
    "EffVsPt_HLT_IsoMu9_EM                          'HLT_IsoMu9                        ; p_{T #mu };       Trigger_Efficiency'  Muon1Pt_EM_HLT_IsoMu9                            Muon1Pt_EM",
    "EffVsEta_HLT_IsoMu9_EM                         'HLT_IsoMu9                        ; #eta_{#mu };      Trigger_Efficiency'  Muon1Eta_EM_HLT_IsoMu9                           Muon1Eta_EM",     
    "EffVsPt_HLT_Ele15_SW_L1R_EM                    'HLT_Ele15_SW_L1R                  ; p_{T e };         Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele15_SW_L1R                  Electron1Pt_EM",
    "EffVsEta_HLT_Ele15_SW_L1R_EM                   'HLT_Ele15_SW_L1R                  ; #eta_{e };        Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele15_SW_L1R                 Electron1Eta_EM",
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R_EM      'HLT_Ele15_SW_LooseTrackIso_L1R    ; p_{T e };         Trigger_Efficiency'  Electron1Pt_EM_HLT_Ele15_SW_LooseTrackIso_L1R    Electron1Pt_EM",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R_EM     'HLT_Ele15_SW_LooseTrackIso_L1R    ; #eta_{e };        Trigger_Efficiency'  Electron1Eta_EM_HLT_Ele15_SW_LooseTrackIso_L1R   Electron1Eta_EM",   
   
    )
      
hltTopPostJet = HLTTopPostProcessor.clone()

hltTopPostJet.subDirs        = ['HLT/Top/Jets']  
hltTopPostJet.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
   #  "EffVsEt_HLT_QuadJet30                          'HLT_QuadJet30                        ; E_{T jet};       Trigger_Efficiency'  Jet1Et_M_HLT_QuadJet30                         Jet1Et_M",
     "Efficiencies_jetTriggers_semimu                  'HLT efficiency wrt acc+offline        ;  ;              Trigger Efficiency'               numer   denom",
   #  "EffVsEta_HLT_QuadJet30                          'HLT_QuadJet30                        ; #eta_{ jet};       Trigger_Efficiency'  Jet1Eta_M_HLT_QuadJet30                         Jet1Eta_M",
     
 
     "Efficiencies_jetTriggers_semiel                   'HLT efficiency wrt acc+offline        ; ;              Trigger Efficiency'               numer_el   denom_el",
    # "EffVsEt_HLT_QuadJet30_el                          'HLT_QuadJet30                        ; E_{T jet};       Trigger_Efficiency'  Jet1Et_E_HLT_QuadJet30                         Jet1Et_E",
    # "EffVsEta_HLT_QuadJet30_el                          'HLT_QuadJet30                        ; #eta_{ jet};       Trigger_Efficiency'  Jet1Eta_E_HLT_QuadJet30                         Jet1Eta_M",
    
    )
          
       
hltTopPost4Jets = HLTTopPostProcessor.clone()

hltTopPost4Jets.subDirs        = ['HLT/Top/4JetsPlus1MuonToCompareWithData']  
hltTopPost4Jets.efficiency     = (
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
   "EffVsPt_HLTMu9_4Jets1MuonMon      'HLT_Mu9    ; p_{T #mu };         Trigger_Efficiency'  Muon1Pt_4Jets1MuonHLTMu9Mon     Muon1Pt_4Jets1MuonMon",
   "EffVsEta_HLTMu9_4Jets1MuonMon     'HLT_Mu9   ; #eta_{#mu };         Trigger_Efficiency'  Muon1Eta_4Jets1MuonHLTMu9Mon    Muon1Eta_4Jets1MuonMon",   
   "EffVsNJets_HLTMu9_4Jets1MuonMon        'HLT_Mu9   ; Jet multiplicity;    Trigger_Efficiency'  NJets_4Jets1MuonHLTMu9Mon       NJets_4Jets1MuonMon",   
                      
    )
          
          
    

    
HLTTopPostVal = cms.Sequence(
    hltTopPostSemimu   *
    hltTopPostSemiel   *
    hltTopPostDimu   *
    hltTopPostDiel   *
    hltTopPostEmu   *
    hltTopPostJet   *
    hltTopPost4Jets
   
)
