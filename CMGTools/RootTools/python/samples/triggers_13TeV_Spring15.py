################## 
## Triggers for HLT_MC_SPRING15 and Run II
## Based on HLT_MC_SPRING15 and /frozen/2015/25ns14e33/v2.1/HLT/V1 and /frozen/2015/50ns_5e33/v2.1/HLT/V5
## Names with _50ns are unprescaled at 50ns but prescaled at 25ns
## Names with _run1 are for comparing Spring15 MC to 8 TeV data: they're the closest thing I could find to run1 triggers, they're prescaled or even excluded in data but should appear in MC.

triggers_mumu_run1   = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_DZ_v*"]
triggers_mumu_iso    = [ "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v*", "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*" ]
triggers_mumu_noniso_50ns = [ "HLT_Mu27_TkMu8_v*" ]
triggers_mumu_noniso = [ "HLT_Mu30_TkMu11_v*" ]
triggers_mumu_ss = [ "HLT_Mu17_Mu8_SameSign_v*" ]
triggers_mumu = triggers_mumu_iso

triggers_ee_run1   = ["HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL*" ]
triggers_ee = [ "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*" ]

triggers_mue_run1   = [ "HLT_Mu17_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*", 
                        "HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v*" ]
triggers_mue   = [ "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v*", 
                   "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v*" ]

triggers_mumu_ht =  [ "HLT_DoubleMu8_Mass8_PFHT300_v*" ]
triggers_ee_ht =  [ "HLT_DoubleEle8_CaloIdM_TrackIdM_Mass8_PFHT300_v*" ]
triggers_mue_ht = [ "HLT_Mu8_Ele8_CaloIdM_TrackIdM_Mass8_PFHT300_v*" ]

triggers_3e = [ "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v*" ]
triggers_3mu = [ "HLT_TripleMu_12_10_5_v*" ]
triggers_3mu_alt = [ "HLT_TrkMu15_DoubleTrkMu5NoFiltersNoVtx_v*" ]
triggers_2mu1e = [ "HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v*" ]
triggers_2e1mu = [ "HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v*" ]

triggers_1mu_iso_r  = [ 'HLT_IsoMu24_eta2p1_v*', 'HLT_IsoTkMu24_eta2p1_v*'  ]
triggers_1mu_iso_w  = [ 'HLT_IsoMu27_v*', 'HLT_IsoTkMu27_v*'  ]
triggers_1mu_iso_r_50ns = [ 'HLT_IsoMu17_eta2p1_v*', 'HLT_IsoTkMu17_eta2p1_v*'  ]
triggers_1mu_iso_w_50ns = [ 'HLT_IsoMu20_v*', 'HLT_IsoTkMu20_v*'  ]
triggers_1mu_noniso = [ 'HLT_Mu45_eta2p1_v*', 'HLT_Mu50_v*' ]
triggers_1mu_iso_50ns = triggers_1mu_iso_r_50ns + triggers_1mu_iso_w_50ns
triggers_1mu_iso      = triggers_1mu_iso_r + triggers_1mu_iso_w

# note: here the WP75 is th name in MC, WPLoose and WPTight should be in data
triggers_1e_50ns = [ "HLT_Ele27_eta2p1_WP75_Gsf_v*", "HLT_Ele27_eta2p1_WPLoose_Gsf_v*", "HLT_Ele27_eta2p1_WPTight_Gsf_v*" ]
triggers_1e      = [ "HLT_Ele32_eta2p1_WP75_Gsf_v*", "HLT_Ele32_eta2p1_WPLoose_Gsf_v*", "HLT_Ele32_eta2p1_WPTight_Gsf_v*" ]

### GP: did not look at anything below this

### ----> for the MT2 analysis

triggers_HT900 = ["HLT_PFHT900_v*"]
triggers_MET170 = ["HLT_PFMET170_NoiseCleaned_v*"]
triggers_HTMET = ["HLT_PFHT350_PFMET120_NoiseCleaned_v*"]

triggers_photon155 = ["HLT_Photon155_v*"]

triggers_MT2_mumu = triggers_mumu_iso
triggers_MT2_ee   = triggers_ee

triggers_MT2_mue = triggers_mue
