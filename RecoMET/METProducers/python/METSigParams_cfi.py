import FWCore.ParameterSet.Config as cms

METSignificance_params = cms.PSet(
    #For METSignificance
    #ECAL:
    EB_EtResPar = cms.vdouble(0.2,0.03,0.005),
    EB_PhiResPar = cms.vdouble(0.00502),    # 0.0174/sqrt(12)
    EE_EtResPar = cms.vdouble(0.2,0.03,0.005),
    EE_PhiResPar = cms.vdouble(0.02511),    # 0.087/sqrt(12)
    #HCAL:
    HB_EtResPar = cms.vdouble(0.,1.22,0.05),
    HB_PhiResPar = cms.vdouble(0.02511),    # 0.087/sqrt(12)
    HE_EtResPar = cms.vdouble(0.,1.3,0.05),
    HE_PhiResPar = cms.vdouble(0.02511),    # 0.087/sqrt(12)
    HO_EtResPar = cms.vdouble(0.,1.3,0.005),
    HO_PhiResPar = cms.vdouble(0.02511),    # 0.087/sqrt(12)
    HF_EtResPar = cms.vdouble(0.,1.82,0.09),
    HF_PhiResPar = cms.vdouble(0.05022),     # 0.174/sqrt(12)

    #PF:
    # type 1: charged hadron - essentially tracking resolutions  (22-Nov-2010: ignored, now using tracking error matrix)
    PF_EtResType1 = cms.vdouble(0.05,0,0),
    PF_PhiResType1 = cms.vdouble(0.002),  

    # type 2: EM with track (electron) - essentially tracking resolution  
    PF_EtResType2 = cms.vdouble(0.05,0,0),
    PF_PhiResType2 = cms.vdouble(0.002),

    #type 3: muon  (22-Nov-2010: ignored, now using tracking error matrix)
    PF_EtResType3 = cms.vdouble(0.05,0,0),
    PF_PhiResType3 = cms.vdouble(0.002),

    # type 4: EM witout track (photon)  (From A. Khukhunaishvili tuning)
    PF_EtResType4 = cms.vdouble(0.042,0.100, 0.),
    PF_PhiResType4 = cms.vdouble(0.0028, 0.0, 0.0022),

    # type 5: hadron without track (all calorimeter)   (From A. Khukhunaishvili tuning)
    PF_EtResType5 = cms.vdouble(0.41,0.52,0.25),
    PF_PhiResType5 = cms.vdouble(0.10, 0.10, 0.13),

    # type 6: hadron without track (Forward HCAL)
    PF_EtResType6 = cms.vdouble(0.,1.22,0.05),
    PF_PhiResType6 = cms.vdouble(0.02511),

    # type 7: EM without track (Forward HCAL)
    PF_EtResType7 = cms.vdouble(0.,1.22,0.05),
    PF_PhiResType7 = cms.vdouble(0.02511),  
    
    # Jet Resolution
    resolutionsEra     = cms.string('Spring10'),
    resolutionsAlgo    = cms.string('AK5PF'),
    ptresolthreshold   = cms.double(10.),
    #temporary rough fix for low pT PFJets
    #10 eta bins
    jdpt0  = cms.vdouble(0.749, 0.829, 1.099, 1.355, 1.584, 1.807, 2.035, 2.217, 2.378, 2.591, ),
    jdphi0 = cms.vdouble(0.034, 0.034, 0.034, 0.034, 0.032, 0.031, 0.028, 0.027, 0.027, 0.027, ),
    jdpt1  = cms.vdouble(0.718, 0.813, 1.133, 1.384, 1.588, 1.841, 2.115, 2.379, 2.508, 2.772, ),
    jdphi1 = cms.vdouble(0.034, 0.035, 0.035, 0.035, 0.035, 0.034, 0.031, 0.030, 0.029, 0.027, ),
    jdpt2  = cms.vdouble(0.841, 0.937, 1.316, 1.605, 1.919, 2.295, 2.562, 2.722, 2.943, 3.293, ),
    jdphi2 = cms.vdouble(0.040, 0.040, 0.040, 0.040, 0.040, 0.038, 0.036, 0.035, 0.034, 0.033, ),
    jdpt3  = cms.vdouble(0.929, 1.040, 1.460, 1.740, 2.042, 2.289, 2.639, 2.837, 2.946, 2.971, ),
    jdphi3 = cms.vdouble(0.042, 0.043, 0.044, 0.043, 0.041, 0.039, 0.039, 0.036, 0.034, 0.031, ),
    jdpt4  = cms.vdouble(0.850, 0.961, 1.337, 1.593, 1.854, 2.005, 2.209, 2.533, 2.812, 3.047, ),
    jdphi4 = cms.vdouble(0.042, 0.042, 0.043, 0.042, 0.038, 0.036, 0.036, 0.033, 0.031, 0.031, ),
    jdpt5  = cms.vdouble(1.049, 1.149, 1.607, 1.869, 2.012, 2.219, 2.289, 2.412, 2.695, 2.865, ),
    jdphi5 = cms.vdouble(0.069, 0.069, 0.064, 0.058, 0.053, 0.049, 0.049, 0.043, 0.039, 0.040, ),
    jdpt6  = cms.vdouble(1.213, 1.298, 1.716, 2.015, 2.191, 2.612, 2.863, 2.879, 2.925, 2.902, ),
    jdphi6 = cms.vdouble(0.084, 0.080, 0.072, 0.065, 0.066, 0.060, 0.051, 0.049, 0.045, 0.045, ),
    jdpt7  = cms.vdouble(1.094, 1.139, 1.436, 1.672, 1.831, 2.050, 2.267, 2.549, 2.785, 2.860, ),
    jdphi7 = cms.vdouble(0.077, 0.072, 0.059, 0.050, 0.045, 0.042, 0.039, 0.039, 0.037, 0.031, ),
    jdpt8  = cms.vdouble(0.889, 0.939, 1.166, 1.365, 1.553, 1.805, 2.060, 2.220, 2.268, 2.247, ),
    jdphi8 = cms.vdouble(0.059, 0.057, 0.051, 0.044, 0.038, 0.035, 0.037, 0.032, 0.028, 0.028, ),
    jdpt9  = cms.vdouble(0.843, 0.885, 1.245, 1.665, 1.944, 1.981, 1.972, 2.875, 3.923, 7.510, ),
    jdphi9 = cms.vdouble(0.062, 0.059, 0.053, 0.047, 0.042, 0.045, 0.036, 0.032, 0.034, 0.044, ),
    )
