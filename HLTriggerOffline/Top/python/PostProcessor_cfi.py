import FWCore.ParameterSet.Config as cms


HLTTopPostProcessor = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring('HLT/Top/'),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(''),
    resolution     = cms.vstring(''),
    efficiency     = cms.vstring(
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffVsPt_HLT_Mu9                                'HLT_Mu9                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_M_HLT_Mu9                                Muon1Pt_M",
    "EffVsEta_HLT_Mu9                               'HLT_Mu9                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_M_HLT_Mu9                               Muon1Eta_M",
    "EffVsPt_HLT_Mu15                               'HLT_Mu15                          ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_M_HLT_Mu15                               Muon1Pt_M",
    "EffVsEta_HLT_Mu15                              'HLT_Mu15                          ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_M_HLT_Mu15                              Muon1Eta_M",
    "EffVsPt_HLT_IsoMu9                             'HLT_IsoMu9                        ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_M_HLT_IsoMu9                             Muon1Pt_M",
    "EffVsEta_HLT_IsoMu9                            'HLT_IsoMu9                        ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_M_HLT_IsoMu9                            Muon1Eta_M",
    "EffVsPt_HLT_Ele15_SW_L1R                       'HLT_Ele15_SW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_E_HLT_Ele15_SW_L1R                   Electron1Pt_E",
    "EffVsEta_HLT_Ele15_SW_L1R                      'HLT_Ele15_SW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_E_HLT_Ele15_SW_L1R                  Electron1Eta_E",
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R         'HLT_Ele15_SW_LooseTrackIso_L1R    ; p_{T e};          Trigger_Efficiency'  Electron1Pt_E_HLT_Ele15_SW_LooseTrackIso_L1R     Electron1Pt_E",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R        'HLT_Ele15_SW_LooseTrackIso_L1R    ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_E_HLT_Ele15_SW_LooseTrackIso_L1R    Electron1Eta_E",    
    "EffVsPt_HLT_Mu9_MM                             'HLT_Mu9                           ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_Mu9                               Muon1Pt_MM",
    "EffVsEta_HLT_Mu9_MM                            'HLT_Mu9                           ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_Mu9                              Muon1Eta_MM",
    "EffVsPt_HLT_Mu15_MM                            'HLT_Mu15                          ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_Mu15                              Muon1Pt_MM",
    "EffVsEta_HLT_Mu15_MM                           'HLT_Mu15                          ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_Mu15                             Muon1Eta_MM",
    "EffVsPt_HLT_IsoMu9_MM                          'HLT_IsoMu9                        ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_IsoMu9                            Muon1Pt_MM",
    "EffVsEta_HLT_IsoMu9_MM                         'HLT_IsoMu9                        ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_IsoMu9                           Muon1Eta_MM",
    "EffVsPt_HLT_DoubleMu3_MM                       'HLT_DoubleMu3                     ; p_{T #mu 1};      Trigger_Efficiency'  Muon1Pt_MM_HLT_DoubleMu3                         Muon1Pt_MM",
    "EffVsEta_HLT_DoubleMu3_MM                      'HLT_DoubleMu3                     ; #eta_{#mu 1};     Trigger_Efficiency'  Muon1Eta_MM_HLT_DoubleMu3                        Muon1Eta_MM",    
    "EffVsPt_HLT_Ele15_SW_L1R_EE                    'HLT_Ele15_SW_L1R                  ; p_{T e 1};        Trigger_Efficiency'  Electron1Pt_EE_HLT_Ele15_SW_L1R                  Electron1Pt_EE",
    "EffVsEta_HLT_Ele15_SW_L1R_EE                   'HLT_Ele15_SW_L1R                  ; #eta_{e 1};       Trigger_Efficiency'  Electron1Eta_EE_HLT_Ele15_SW_L1R                 Electron1Eta_EE",
    "EffVsPt_HLT_Ele15_SW_LooseTrackIso_L1R_EE      'HLT_Ele15_SW_LooseTrackIso_L1R    ; p_{T e 1};        Trigger_Efficiency'  Electron1Pt_EE_HLT_Ele15_SW_LooseTrackIso_L1R    Electron1Pt_EE",
    "EffVsEta_HLT_Ele15_SW_LooseTrackIso_L1R_EE     'HLT_Ele15_SW_LooseTrackIso_L1R    ; #eta_{e 1};       Trigger_Efficiency'  Electron1Eta_EE_HLT_Ele15_SW_LooseTrackIso_L1R   Electron1Eta_EE",   
    "EffVsPt_HLT_DoubleEle10_SW_L1R_EE              'HLT_DoubleEle10_SW_L1R            ; p_{T e 1};        Trigger_Efficiency'  Electron1Pt_EE_HLT_DoubleEle10_SW_L1R            Electron1Pt_EE",
    "EffVsEta_HLT_DoubleEle10_SW_L1R_EE             'HLT_DoubleEle10_SW_L1R            ; #eta_{e 1};       Trigger_Efficiency'  Electron1Eta_EE_HLT_DoubleEle10_SW_L1R           Electron1Eta_EE",    
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
    #####
    "Efficiencies_muontriggers                      'HLT efficiency wrt offline        ; eff ;              Trigger Efficiency'               muHLT   MuonEvents",
    
    "Efficiencies_muontriggers_gen                  'HLT efficiency wrt mc             ; eff ;              Trigger Efficiency'               genmuHLT   genMuonEvents",
    ####
    
    
    "Efficiencies_jettriggers_semimu                  'HLT efficiency wrt offline             ; eff ;              Trigger Efficiency'               numer   denom",
    
    "Efficiencies_electrontriggers                      'HLT efficiency wrt offline        ; eff ;              Trigger Efficiency'               elHLT   ElectronEvents",
    
    "Efficiencies_electrontriggers_gen                  'HLT efficiency wrt mc             ; eff ;              Trigger Efficiency'               genelHLT   genElectronEvents"
    ####
    
    )
)

