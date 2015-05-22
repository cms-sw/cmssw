################## Triggers (FIXME: update to the PHYS14 Trigger Menu)


triggers_mumu_run1   = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_v*"]
triggers_mumu_iso    = [ "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v*", "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v*" ]
triggers_mumu_noniso = [ "HLT_Mu30_TkMu11_v*" ]
triggers_mumu = triggers_mumu_iso + triggers_mumu_noniso

triggers_ee_run1   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                      "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                      "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v*"]
triggers_ee = [ "HLT_Ele23_Ele12_CaloId_TrackId_Iso_v*" ]
triggers_3e = [ "HLT_Ele17_Ele12_Ele10_CaloId_TrackId_v*" ]
triggers_mue   = [
    "HLT_Mu23_TrkIsoVVL_Ele12_Gsf_CaloId_TrackId_Iso_MediumWP_v*",
    "HLT_Mu8_TrkIsoVVL_Ele23_Gsf_CaloId_TrackId_Iso_MediumWP_v*"
    ]

triggers_multilep  = triggers_mumu + triggers_ee + triggers_3e + triggers_mue

triggers_1mu_iso    = [ 'HLT_IsoMu24_eta2p1_IterTrk02_v*', 'HLT_IsoTkMu24_eta2p1_IterTrk02_v*'  ]
triggers_1mu_isowid = [ 'HLT_IsoMu24_IterTrk02_v*', 'HLT_IsoTkMu24_IterTrk02_v*'  ]
triggers_1mu_isolow = [ 'HLT_IsoMu20_eta2p1_IterTrk02_v*', 'HLT_IsoTkMu20_eta2p1_IterTrk02_v*'  ]
triggers_1mu_noniso = [ 'HLT_Mu40_v*' ]
triggers_1mu = triggers_1mu_iso  + triggers_1mu_isowid + triggers_1mu_isolow + triggers_1mu_noniso

triggers_1e = [ "HLT_Ele27_eta2p1_WP85_Gsf_v*", "HLT_Ele32_eta2p1_WP85_Gsf_v*" ]

triggersFR_1mu  = [ 'HLT_Mu5_v*', 'HLT_RelIso1p0Mu5_v*', 'HLT_Mu12_v*', 'HLT_Mu24_eta2p1_v*', 'HLT_Mu40_eta2p1_v*' ]
triggersFR_mumu = [ 'HLT_Mu17_Mu8_v*', 'HLT_Mu17_TkMu8_v*', 'HLT_Mu8_v*', 'HLT_Mu17_v*' ]
triggersFR_1e   = [ 'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*', 'HLT_Ele17_CaloIdL_CaloIsoVL_v*', 'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*', 'HLT_Ele8__CaloIdL_CaloIsoVL_v*']
triggersFR_mue  = triggers_mue[:]
triggersFR_MC = triggersFR_1mu + triggersFR_mumu + triggersFR_1e + triggersFR_mue



### ----> for the MT2 analysis

triggers_HT900 = ["HLT_PFHT900_v*"]
triggers_MET170 = ["HLT_PFMET170_NoiseCleaned_v*"]
triggers_HTMET = ["HLT_PFHT350_PFMET120_NoiseCleaned_v*"]

triggers_photon155 = ["HLT_Photon155_v*"]

triggers_MT2_mumu = triggers_mumu_iso
triggers_MT2_ee   = triggers_ee

triggers_MT2_mue = triggers_mue
