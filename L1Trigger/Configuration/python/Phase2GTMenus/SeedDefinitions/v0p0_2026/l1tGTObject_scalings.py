### L1T Object threshold scalings 
### as derived by the Phase-2 L1 DPG Menu team
### using the Phase-2 MenuTools: https://github.com/cms-l1-dpg/Phase2-L1MenuTools

### Corresponds to version V50 derived with the Annual Review 2026 config in the 17_0_x release (CMSSW_17_0_0_pre6)
### See the L1T Phase-2 Menu Twiki for more details: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PhaseIIL1TriggerMenuTools#Phase_2_L1_Trigger_objects_based

### NB Objects starting with L1 are not yet part of the Step1 Menu and thus no seeds implemented in the P2GT emulator

scalings = {
    "CL2Electrons": {
        "Iso": {
            "barrel": {
                "offset": 1.97,
                "slope": 1.12
            },
            "endcap": {
                "offset": 2.04,
                "slope": 1.24
            }
        },
        "NoIso": {
            "barrel": {
                "offset": 2.02,
                "slope": 1.13
            },
            "endcap": {
                "offset": 2.59,
                "slope": 1.25
            }
        }
    },
    "CL2EtSum": {
        "default": {
            "inclusive": {
                "offset": 32.9,
                "slope": 1.69
            }
        }
    },
    "CL2HtSum": {
        "HT": {
            "inclusive": {
                "offset": 42.92,
                "slope": 1.06
            }
        },
        "MHT": {
            "inclusive": {
                "offset": -15.71,
                "slope": 1.14
            }
        }
    },
    "CL2JetsSC4": {
        "default": {
            "barrel": {
                "offset": 21.2,
                "slope": 1.18
            },
            "endcap": {
                "offset": 12.21,
                "slope": 1.56
            },
            "forwardHF": {
                "offset": -6.35,
                "slope": 1.34
            },
            "forwardHGC": {
                "offset": 48.12,
                "slope": 1.22
            }
        }
    },
    "CL2JetsSC8": {
        "default": {
            "barrel": {
                "offset": 39.66,
                "slope": 1.24
            },
            "endcap": {
                "offset": 30.25,
                "slope": 1.89
            },
            "forwardHF": {
                "offset": -9.34,
                "slope": 1.82
            },
            "forwardHGC": {
                "offset": 61.95,
                "slope": 1.32
            }
        }
    },
    "CL2Photons": {
        "Iso": {
            "barrel": {
                "offset": 3.42,
                "slope": 1.08
            },
            "endcap": {
                "offset": 7.26,
                "slope": 1.01
            }
        },
        "NoIso": {
            "barrel": {
                "offset": 4.76,
                "slope": 1.06
            },
            "endcap": {
                "offset": 6.15,
                "slope": 1.07
            }
        }
    },
    "CL2Taus": {
        "default": {
            "barrel": {
                "offset": 2.21,
                "slope": 1.27
            },
            "endcap": {
                "offset": -6.85,
                "slope": 1.92
            }
        },
        "extended": {
            "barrel": {
                "offset": 2.21,
                "slope": 1.27
            },
            "endcap": {
                "offset": -7.41,
                "slope": 1.89
            }
        }
    },
    "GMTSaPromptMuons": {
        "default": {
            "barrel": {
                "offset": 1.07,
                "slope": 1.69
            },
            "endcap": {
                "offset": -3.1,
                "slope": 1.21
            },
            "overlap": {
                "offset": -0.69,
                "slope": 1.48
            }
        }
    },
    "GMTTkMuons": {
        "Dynamic": {
            "barrel": {
                "offset": 1.0,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.76,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.07,
                "slope": 1.03
            }
        },
        "Loose": {
            "barrel": {
                "offset": 1.0,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.78,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.07,
                "slope": 1.04
            }
        },
        "Medium": {
            "barrel": {
                "offset": 1.0,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.78,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.07,
                "slope": 1.04
            }
        },
        "Tight": {
            "barrel": {
                "offset": 1.01,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.77,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.06,
                "slope": 1.04
            }
        },
        "VLoose": {
            "barrel": {
                "offset": 1.01,
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
        "default": {
            "barrel": {
                "offset": 1.0,
                "slope": 1.04
            },
            "endcap": {
                "offset": 0.76,
                "slope": 1.04
            },
            "overlap": {
                "offset": 1.07,
                "slope": 1.03
            }
        }
    },
    "L1EG": {
        "default": {
            "barrel": {
                "offset": 4.88,
                "slope": 1.15
            },
            "endcap": {
                "offset": 2.29,
                "slope": 1.26
            }
        },
        "extended": {
            "barrel": {
                "offset": 4.88,
                "slope": 1.15
            },
            "endcap": {
                "offset": 2.28,
                "slope": 1.22
            }
        }
    },
    "L1TrackHT": {
        "HT": {
            "inclusive": {
                "offset": 19.11,
                "slope": 2.61
            }
        },
        "MHT": {
            "inclusive": {
                "offset": -15.11,
                "slope": 2.2
            }
        }
    },
    "L1TrackJet": {
        "default": {
            "barrel": {
                "offset": -30.75,
                "slope": 6.18
            },
            "endcap": {
                "offset": -20.27,
                "slope": 6.4
            }
        }
    },
    "L1TrackMET": {
        "default": {
            "inclusive": {
                "offset": -81.17,
                "slope": 8.89
            }
        }
    },
    "L1caloJet": {
        "default": {
            "barrel": {
                "offset": -26.55,
                "slope": 2.18
            },
            "endcap": {
                "offset": 24.21,
                "slope": 2.0
            },
            "forwardHF": {
                "offset": -30.34,
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
                "offset": -9.82,
                "slope": 1.71
            },
            "endcap": {
                "offset": -49.41,
                "slope": 2.67
            }
        },
        "extended": {
            "barrel": {
                "offset": -9.82,
                "slope": 1.71
            },
            "endcap": {
                "offset": -54.9,
                "slope": 2.77
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
                "offset": 2.1,
                "slope": 1.53
            },
            "endcap": {
                "offset": 14.32,
                "slope": 1.51
            },
            "forwardHF": {
                "offset": -6.35,
                "slope": 1.34
            },
            "forwardHGC": {
                "offset": 47.78,
                "slope": 1.22
            }
        }
    },
    "L1puppiJetSC4NG": {
        "default": {
            "barrel": {
                "offset": 10.67,
                "slope": 1.38
            },
            "endcap": {
                "offset": 23.12,
                "slope": 1.49
            }
        }
    },
    "L1puppiMLMET": {
        "default": {
            "inclusive": {
                "offset": 25.45,
                "slope": 1.62
            }
        }
    }
}
