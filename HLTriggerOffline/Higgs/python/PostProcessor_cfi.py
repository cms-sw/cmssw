import FWCore.ParameterSet.Config as cms

HLTHiggsPostProcessor = cms.EDAnalyzer("DQMGenericClient",
    subDirs           = cms.untracked.vstring('HLT/Higgs/HWW'),
    verbose           = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName    = cms.untracked.string(''),
    commands          = cms.vstring(''),
    resolution        = cms.vstring(''),
    efficiency        = cms.vstring(
  #  "TrigEFF                      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
   # "EffVsPt_HLT_Mu3                                'HLT_Mu3                           ; p_{T #mu};        Trigger_Efficiency'  Muon1Pt_HLT_Mu3                                Muon1Pt",
    #"EffVsEta_HLT_Mu3                               'HLT_Mu3                           ; #eta_{#mu};       Trigger_Efficiency'  Muon1Eta_HLT_Mu3                               Muon1Eta",
   
   
   # "EffVsPt_HLT_Ele10_LW_L1R                       'HLT_Ele10_LW_L1R                  ; p_{T e};          Trigger_Efficiency'  Electron1Pt_HLT_Ele10_LW_L1R                   Electron1Pt",
   # "EffVsEta_HLT_Ele10_LW_L1R                      'HLT_Ele10_LW_L1R                  ; #eta_{e};         Trigger_Efficiency'  Electron1Eta_HLT_Ele10_LW_L1R                  Electron1Eta",
   
   # "EffVsPt_HLT_Photon15_L1R                                'HLT_Photon15_L1R                           ; p_{T #gamma};        Trigger_Efficiency'  Photon1Pt_HLT_Photon15_L1R                                Photon1Pt",
   # "EffVsEta_HLT_Photon15_L1R                               'HLT_Photon15_L1R                           ; #eta_{#gamma};       Trigger_Efficiency'  Photon1Eta_HLT_Photon15_L1R                               Photon1Eta",
   
   #"EffVsPt_HLT_Photon15_TrackIso_L1R                                'HLT_Photon15_TrackIso_L1R                           ; p_{T #gamma};        Trigger_Efficiency'  Photon1Pt_HLT_Photon15_TrackIso_L1R                                Photon1Pt",
   # "EffVsEta_HLT_Photon15_TrackIso_L1R                               'HLT_Photon15_TrackIso_L1R                           ; #eta_{#gamma};       Trigger_Efficiency'  Photon1Eta_HLT_Photon15_TrackIso_L1R                               Photon1Eta",
   
   #"EffVsPt_HLT_DoublePhoton10_L1R                                'HLT_DoublePhoton10_L1R                           ; p_{T #gamma};        Trigger_Efficiency'  Photon1Pt_HLT_DoublePhoton10_L1R                                Photon1Pt",
   # "EffVsEta_HLT_DoublePhoton10_L1R                               'HLT_DoublePhoton10_L1R                           ; #eta_{#gamma};       Trigger_Efficiency'  Photon1Eta_HLT_DoublePhoton10_L1R                               Photon1Eta",
   
   
    
    )
)

