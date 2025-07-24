### L1T Object threshold scalings 
### as derived by the Phase-2 L1 DPG Menu team
### using the Phase-2 MenuTools: https://github.com/cms-l1-dpg/Phase2-L1MenuTools

### Corresponds to version v49 derived with the Annual Review 2025 config in the 15_1_x release
### See the L1T Phase-2 Menu Twiki for more details: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PhaseIIL1TriggerMenuTools#Phase_2_L1_Trigger_objects_based

### NB Objects starting with L1 are not yet part of the Step1 Menu and thus no seeds implemented in the P2GT emulator

scalings = {
    "CL2Electrons": {
        "Iso": {
            "barrel": {
                "offset": 3.58,
                "slope": 1.17
            },
            "endcap": {
                "offset": 2.05,
                "slope": 1.24
            }
        },
        "NoIso": {
            "barrel": {
                "offset": 3.69,
                "slope": 1.18
            },
            "endcap": {
                "offset": 2.6,
                "slope": 1.25
            }
        }
    },
    "CL2EtSum": {
        "default": {
            "inclusive": {
                "offset": 36.8,
                "slope": 1.68
            }
        }
    },
    "CL2HtSum": {
        "HT": {
            "inclusive": {
                "offset": 37.73,
                "slope": 1.17
            }
        },
        "MHT": {
            "inclusive": {
                "offset": -13.67,
                "slope": 1.2
            }
        }
    },
    "CL2JetsSC4": {
        "default": {
            "barrel": {
                "offset": 16.52,
                "slope": 1.36
            },
            "endcap": {
                "offset": 10.14,
                "slope": 1.59
            },
            "forwardHF": {
                "offset": -6.35,
                "slope": 1.34
            },
            "forwardHGC": {
                "offset": 47.81,
                "slope": 1.22
            }
        }
    },
    "CL2JetsSC8": {
        "default": {
            "barrel": {
                "offset": 25.88,
                "slope": 1.4
            },
            "endcap": {
                "offset": 24.88,
                "slope": 1.46
            },
            "forwardHF": {
                "offset": -5.31,
                "slope": 1.5
            },
            "forwardHGC": {
                "offset": 63.24,
                "slope": 1.28
            }
        }
    },
    "CL2Photons": {
        "Iso": {
            "barrel": {
                "offset": 3.43,
                "slope": 1.13
            },
            "endcap": {
                "offset": 7.71,
                "slope": 0.97
            }
        },
        "NoIso": {
            "barrel": {
                "offset": 4.44,
                "slope": 1.12
            },
            "endcap": {
                "offset": 6.01,
                "slope": 1.06
            }
        }
    },
    "CL2Taus": {
        "default": {
            "barrel": {
                "offset": -0.31,
                "slope": 1.38
            },
            "endcap": {
                "offset": -4.96,
                "slope": 1.89
            }
        }
    },
    "GMTSaPromptMuons": {
        "default": {
            "barrel": {
                "offset": 1.1,
                "slope": 1.69
            },
            "endcap": {
                "offset": -3.1,
                "slope": 1.21
            },
            "overlap": {
                "offset": -1.19,
                "slope": 1.32
            }
        }
    },
    "GMTTkMuons": {
        "Loose": {
            "barrel": {
                "offset": 1.0,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.81,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.09,
                "slope": 1.03
            }
        },
        "Medium": {
            "barrel": {
                "offset": 1.0,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.81,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.08,
                "slope": 1.03
            }
        },
        "Tight": {
            "barrel": {
                "offset": 1.01,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.81,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.07,
                "slope": 1.04
            }
        },
        "VLoose": {
            "barrel": {
                "offset": 1.01,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.85,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.08,
                "slope": 1.03
            }
        },
        "default": {
            "barrel": {
                "offset": 1.0,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.79,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.09,
                "slope": 1.03
            }
        }
    },
    "L1EG": {
        "default": {
            "barrel": {
                "offset": 4.91,
                "slope": 1.15
            },
            "endcap": {
                "offset": 2.38,
                "slope": 1.24
            }
        }
    },
    "L1TrackHT": {
        "HT": {
            "inclusive": {
                "offset": -36.76,
                "slope": 2.51
            }
        },
        "MHT": {
            "inclusive": {
                "offset": -9.98,
                "slope": 2.16
            }
        }
    },
    "L1TrackJet": {
        "default": {
            "barrel": {
                "offset": 13.45,
                "slope": 5.26
            },
            "endcap": {
                "offset": 17.38,
                "slope": 7.67
            }
        }
    },
    "L1TrackMET": {
        "default": {
            "inclusive": {
                "offset": -77.78,
                "slope": 8.69
            }
        }
    },
    "L1caloJet": {
        "default": {
            "barrel": {
                "offset": 2.83,
                "slope": 1.46
            },
            "endcap": {
                "offset": 25.75,
                "slope": 1.99
            },
            "forwardHF": {
                "offset": -30.41,
                "slope": 1.57
            },
            "forwardHGC": {
                "offset": -46.91,
                "slope": 2.4
            }
        }
    },
    "L1caloTau": {
        "default": {
            "barrel": {
                "offset": -8.82,
                "slope": 1.63
            },
            "endcap": {
                "offset": -49.41,
                "slope": 2.67
            }
        }
    },
    "L1hpsTau": {
        "default": {
            "barrel": {
                "offset": -6.51,
                "slope": 1.85
            },
            "endcap": {
                "offset": 15.2,
                "slope": 1.75
            }
        }
    },
    "L1nnCaloTau": {
        "default": {
            "barrel": {
                "offset": -1.73,
                "slope": 1.33
            },
            "endcap": {
                "offset": -6.3,
                "slope": 1.4
            }
        }
    },
    "L1puppiExtJetSC4": {
        "default": {
            "barrel": {
                "offset": 3.29,
                "slope": 1.61
            },
            "endcap": {
                "offset": 12.25,
                "slope": 1.53
            },
            "forwardHF": {
                "offset": -6.35,
                "slope": 1.34
            },
            "forwardHGC": {
                "offset": 47.57,
                "slope": 1.23
            }
        }
    },
    "L1puppiHistoJetSums": {
        "MHT": {
            "inclusive": {
                "offset": -17.4,
                "slope": 1.23
            }
        }
    },
    "L1puppiJetSC4NG": {
        "default": {
            "barrel": {
                "offset": 10.8,
                "slope": 1.66
            },
            "endcap": {
                "offset": 4.32,
                "slope": 2.15
            }
        }
    },
    "L1puppiMLMET": {
        "default": {
            "inclusive": {
                "offset": 28.82,
                "slope": 1.6
            }
        }
    }
}
