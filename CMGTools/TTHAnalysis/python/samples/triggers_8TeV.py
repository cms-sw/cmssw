triggers_mumu_8TeV = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_v*"]
triggers_ee_8TeV   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                 "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                 "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v*"]

triggers_mue_8TeV   = [
    "HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
    "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*"
    ]

triggersMC_mumu_8TeV = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_v*"]

triggersMC_ee_8TeV   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                   "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                   "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v*"]

triggersMC_mue_8TeV   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                    "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                    "HLT_Ele15_Ele8_Ele5_CaloIdL_TrkIdVL_v*",
                    "HLT_Mu17_Mu8_v*",
                    "HLT_Mu17_TkMu8_v*",
                    "HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                    "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*"
                   ]

triggers_1mu_8TeV = [ 'HLT_IsoMu24_eta2p1_v*' ]
triggersMC_1mu_8TeV  = triggers_1mu_8TeV;
triggersFR_1mu_8TeV  = [ 'HLT_Mu5_v*', 'HLT_RelIso1p0Mu5_v*', 'HLT_Mu12_v*', 'HLT_Mu24_eta2p1_v*', 'HLT_Mu40_eta2p1_v*' ]
triggersFR_mumu_8TeV = [ 'HLT_Mu17_Mu8_v*', 'HLT_Mu17_TkMu8_v*', 'HLT_Mu8_v*', 'HLT_Mu17_v*' ]
triggersFR_1e_8TeV   = [ 'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*', 'HLT_Ele17_CaloIdL_CaloIsoVL_v*', 'HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*', 'HLT_Ele8__CaloIdL_CaloIsoVL_v*']
triggersFR_mue_8TeV  = triggers_mue_8TeV[:]
triggersFR_MC_8TeV = triggersFR_1mu_8TeV + triggersFR_mumu_8TeV + triggersFR_1e_8TeV + triggersFR_mue_8TeV


### ----> for the SUS-13-007

triggers_1muHT_8TeV = ["HLT_PFHT350_Mu15_PFMET45_v*","HLT_PFHT350_Mu15_PFMET50_v*",
                  "HLT_PFNoPUHT350_Mu15_PFMET_45_v*","HLT_PFHT350_Mu15_PFMET50_v*",
                  "HLT_PFHT400_Mu5_PFMET45_v*","HLT_PFHT400_Mu5_PFMET50_v*",
                  "HLT_PFNoPUHT400_Mu5_PFMET45_v*","HLT_PFNoPUHT400_Mu5_PFMET50_v*",
                  "HLT_Mu40_PFHT350_v*","HLT_Mu60_PFHT350_v*",
                  "HLT_Mu40_PFNoPUHT350_v*","HLT_Mu60_PFNoPUHT350_v*"
                  ]

triggers_1eleHT_8TeV = ["HLT_CleanPFHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v*","HLT_CleanPFHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v*",
                   "HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v*","HLT_CleanPFNoPUHT350_Ele5_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET50_v*",
                   "HLT_CleanPFHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v*","HLT_CleanPFHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET60_v*",
                   "HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET45_v*","HLT_CleanPFNoPUHT300_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_PFMET60_v*",
                   "HLT_CleanPFHT300_Ele40_CaloIdVT_CaloIsoVL_TrkIdT_v*","HLT_CleanPFHT300_Ele60_CaloIdVT_CaloIsoVL_TrkIdT_v*",                   
                   "HLT_CleanPFNoPUHT300_Ele40_CaloIdVT_CaloIsoVL_TrkIdT_ v*","HLT_CleanPFNoPUHT300_Ele60_CaloIdVT_CaloIsoVL_TrkIdT_ v*"
                   ]


### ----> for the SUS-13-019

triggers_HT650_8TeV = ["HLT_PFHT650_v*","HLT_PFNoPUHT650_v*"]
triggers_MET150_8TeV = ["HLT_PFMET150_v*"]
triggers_HTMET_8TeV = ["HLT_PFHT350_PFMET100_v*","HLT_PFNoPUHT350_PFMET100_v*"]

triggers_MT2_mumu_8TeV = ["HLT_Mu17_Mu8_v*","HLT_Mu17_TkMu8_v*"]
triggers_MT2_ee_8TeV   = ["HLT_Ele17_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_Ele8_CaloIdT_TrkIdVL_CaloIsoVL_TrkIsoVL_v*",
                     "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*"]

triggers_MT2_mue_8TeV = ["HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*",
                    "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*"
                    ]


### ----> RA1 2012 triggers

triggers_RA1_Bulk_8TeV = [
    "HLT_HT200_v*",
    "HLT_HT250_v*",
    "HLT_HT300_v*",
    "HLT_HT350_v*",
    "HLT_HT450_v*",
    "HLT_HT550_v*",
    "HLT_HT650_v*",
    "HLT_HT750_v*",
    ]
triggers_RA1_Parked_8TeV = [
    "HLT_HT200_AlphaT0p57_v*",
    "HLT_HT300_AlphaT0p53_v*",
    "HLT_HT350_AlphaT0p52_v*",
    "HLT_HT400_AlphaT0p51_v*",
    ]
triggers_RA1_Prompt_8TeV = [
    "HLT_HT250_AlphaT0p55_v*",
    "HLT_HT300_AlphaT0p53_v*",
    "HLT_HT350_AlphaT0p52_v*",
    "HLT_HT400_AlphaT0p51_v*",
    "HLT_HT350_AlphaT0p52_v*",
    ]
triggers_RA1_Single_Mu_8TeV = ["HLT_IsoMu24_eta2p1_v*"]
triggers_RA1_Photon_8TeV    = ["HLT_Photon150_v%d"%i for i in range(1,20)] + ["HLT_Photon160_v%d"%i for i in range(1,20)]
triggers_RA1_Muon_8TeV      = ["HLT_IsoMu24_eta2p1_v%d"%i for i in range(1,20)]


