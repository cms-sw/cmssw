################## Triggers   

# signal triggers from https://twiki.cern.ch/twiki/bin/view/CMS/Monojet
triggers_default = ["HLT_PFMETNoMu90_NoiseCleaned_PFMHTNoMu90_IDTight",
                    "HLT_PFMETNoMu120_NoiseCleaned_PFMHTNoMu120_IDTight"]

triggers_backup = ["HLT_PFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight",
                   "HLT_PFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight",
                   "HLT_PFMET170",
                   "HLT_CaloMET200"]

triggers_gamma = ["HLT_Photon165_HE10",
                  "HLT_Photon175"]

triggers_monojet = triggers_default + triggers_backup + triggers_gamma
