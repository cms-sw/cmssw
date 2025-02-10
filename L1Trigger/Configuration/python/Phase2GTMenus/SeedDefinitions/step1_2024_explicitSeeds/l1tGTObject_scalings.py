### L1T Object threshold scalings 
### as derived by the Phase-2 L1 DPG Menu team
### using the Phase-2 MenuTools: https://github.com/cms-l1-dpg/Phase2-L1MenuTools

### Corresponds to version v44 derived with the Annual Review 2024 config in the 14_2_x release
### See the L1T Phase-2 Menu Twiki for more details: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PhaseIIL1TriggerMenuTools#Phase_2_L1_Trigger_objects_based

### NB Objects starting with L1 are not yet part of the Step1 Menu and thus no seeds implemented in the P2GT emulator

scalings = {
    "CL2Electrons": {
        "Iso": {
            "barrel": {
                "offset": 1.16,
                "slope": 1.18
            },
            "endcap": {
                "offset": 0.18,
                "slope": 1.25
            }
        },
        "NoIso": {
            "barrel": {
                "offset": 1.24,
                "slope": 1.18
            },
            "endcap": {
                "offset": 0.63,
                "slope": 1.25
            }
        }
    },
    "CL2EtSum": {
        "default": {
            "inclusive": {
                "offset": 37.05,
                "slope": 1.64
            }
        }
    },
    "CL2HtSum": {
        "HT": {
            "inclusive": {
                "offset": 45.7,
                "slope": 1.12
            }
        },
        "MHT": {
            "inclusive": {
                "offset": -12.93,
                "slope": 1.16
            }
        }
    },
    "CL2JetsSC4": {
        "default": {
            "barrel": {
                "offset": 17.33,
                "slope": 1.28
            },
            "endcap": {
                "offset": 15.33,
                "slope": 1.67
            },
            "forward": {
                "offset": 71.45,
                "slope": 1.14
            }
        }
    },
    "CL2JetsSC8": {
        "default": {
            "barrel": {
                "offset": 23.98,
                "slope": 1.37
            },
            "endcap": {
                "offset": 28.95,
                "slope": 1.56
            },
            "forward": {
                "offset": 69.06,
                "slope": 1.42
            }
        }
    },
    "CL2Photons": {
        "Iso": {
            "barrel": {
                "offset": 3.04,
                "slope": 1.09
            },
            "endcap": {
                "offset": 7.73,
                "slope": 0.96
            }
        },
        "NoIso": {
            "barrel": {
                "offset": 4.43,
                "slope": 1.07
            },
            "endcap": {
                "offset": 5.22,
                "slope": 1.07
            }
        }
    },
    "CL2Taus": {
        "default": {
            "barrel": {
                "offset": 3.53,
                "slope": 1.26
            },
            "endcap": {
                "offset": -3.15,
                "slope": 1.66
            }
        }
    },
    "GMTSaPromptMuons": {
        "default": {
            "barrel": {
                "offset": 1.08,
                "slope": 1.69
            },
            "endcap": {
                "offset": -2.97,
                "slope": 1.21
            },
            "overlap": {
                "offset": -1.17,
                "slope": 1.35
            }
        }
    },
    "GMTTkMuons": {
        "Loose": {
            "barrel": {
                "offset": 0.96,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.87,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.16,
                "slope": 1.03
            }
        },
        "Medium": {
            "barrel": {
                "offset": 0.95,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.87,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.16,
                "slope": 1.03
            }
        },
        "Tight": {
            "barrel": {
                "offset": 0.94,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.87,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.17,
                "slope": 1.03
            }
        },
        "VLoose": {
            "barrel": {
                "offset": 0.96,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.94,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.17,
                "slope": 1.03
            }
        },
        "default": {
            "barrel": {
                "offset": 0.96,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.87,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.16,
                "slope": 1.03
            }
        }
    },
    "L1EG": {
        "default": {
            "barrel": {
                "offset": 4.36,
                "slope": 1.12
            },
            "endcap": {
                "offset": 5.22,
                "slope": 1.07
            }
        }
    },
    "L1TrackHT": {
        "HT": {
            "inclusive": {
                "offset": -51.83,
                "slope": 2.58
            }
        },
        "MHT": {
            "inclusive": {
                "offset": -19.41,
                "slope": 2.27
            }
        }
    },
    "L1TrackJet": {
        "default": {
            "barrel": {
                "offset": 14.84,
                "slope": 5.15
            },
            "endcap": {
                "offset": 30.24,
                "slope": 7.12
            }
        }
    },
    "L1TrackMET": {
        "default": {
            "inclusive": {
                "offset": -75.85,
                "slope": 8.69
            }
        }
    },
    "L1caloJet": {
        "default": {
            "barrel": {
                "offset": 2.27,
                "slope": 1.48
            },
            "endcap": {
                "offset": 77.98,
                "slope": 1.74
            },
            "forward": {
                "offset": 223.84,
                "slope": 0.89
            }
        }
    },
    "L1caloTau": {
        "default": {
            "barrel": {
                "offset": -10.87,
                "slope": 1.69
            },
            "endcap": {
                "offset": -45.77,
                "slope": 2.54
            }
        }
    },
    "L1hpsTau": {
        "default": {
            "barrel": {
                "offset": 1.88,
                "slope": 1.74
            },
            "endcap": {
                "offset": 37.49,
                "slope": 1.5
            }
        }
    },
    "L1nnCaloTau": {
        "default": {
            "barrel": {
                "offset": -0.72,
                "slope": 1.31
            },
            "endcap": {
                "offset": -6.02,
                "slope": 1.38
            }
        }
    },
    "L1puppiHistoJetSums": {
        "MHT": {
            "inclusive": {
                "offset": -15.69,
                "slope": 1.18
            }
        }
    },
    "L1puppiMLMET": {
        "default": {
            "inclusive": {
                "offset": 29.35,
                "slope": 1.56
            }
        }
    }
}