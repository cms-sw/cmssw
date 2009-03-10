import FWCore.ParameterSet.Config as cms

HLTTopPostVal = cms.EDFilter("PostProcessor",
    subDirs        = cms.untracked.vstring('HLT/Top/'),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    outputFileName = cms.untracked.string(''),
    commands       = cms.vstring(''),
    resolution     = cms.vstring(''),
    efficiency     = cms.vstring(
  #  "TrigEFF      'my title; my x-label;    my y-label' pt_trig_off_mu pt_off_mu",
    "EffPt_Mu15                'HLT_Mu15                   ; p_{T #mu};    Trigger_Efficiency'  pt_trig_off_mu     pt_off_mu",
    "EffEta_Mu15               'HLT_Mu15                   ; #eta_{#mu};   Trigger Efficiency'  eta_trig_off_mu    eta_off_mu",
    "EffPt_IsoEle18            'HLT_IsoEle18_L1R           ; p_{T e}   ;   Trigger Efficiency'  pt_trig_off_el     pt_off_el",
    "EffEta_IsoEle18           'HLT_IsoEle18_L1R           ; #eta_{e}  ;   Trigger Efficiency'  eta_trig_off_el    eta_off_el",
    "EffPt_LooseIsoEle15       'HLT_LooseIsoEle15_LW_L1R   ; p_{T e}   ;   Trigger Efficiency'  pt_trig_off_el_li  pt_off_el",
    "EffEta_LooseIsoEle15      'HLT_LooseIsoEle15_LW_L1R   ; #eta_{e}  ;   Trigger Efficiency'  eta_trig_off_el_li eta_off_el",
    "EffPt_el_Ele15            'HLT_Ele15_SW_L1R           ; p_{T e}   ;   Trigger Efficiency'  pt_trig_off_el_ni  pt_off_el",
    "EffEta_el_Ele15           'HLT_Ele15_SW_L1R           ; #eta_{e}  ;   Trigger Efficiency'  eta_trig_off_el_ni eta_off_el",
  
    
    )
)

