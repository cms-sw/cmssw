### L1T Object ID and Isolation criteria 
### as provided by the L1 DPG object contacts
### in agreement with the subsystems

objectIDs = {
    "CL2Taus": {
        "default": {
            "qual" : 225
        }
    },
    "CL2Photons":{
        "Iso": {
            "qual": {
                "barrel": 0b0010,
                "endcap": 0b0100,
            },
            "iso": {
                "barrel": 0.25,
                "endcap": 0.205,
            }
        }
    },
    "CL2Electrons":{
        "Iso": {
            # "qual": {
            #     "barrel": 0b0010,
            #     "endcap": 0b0010,
            # },
            "iso": {
                "barrel": 0.13,
                "endcap": 0.28,
            }
        },
        "NoIso": {
            "qual": {
                "barrel": 0b0010,
                "endcap": 0b0010,
            },
        },
        "NoIsoLowPt": {
            "qual": {
                "barrel": 0b0010,
                "endcap": 0b0000,
            },
        }
    },
    "GMTTkMuons":{
        "VLoose": {
            "qual": 0b0001,
        },
        "Loose": {
            "qual": 0b0010,
        },
        "Medium": {
            "qual": 0b0100,
        },
        "Tight": {
            "qual": 0b1000,
        },
    }
}