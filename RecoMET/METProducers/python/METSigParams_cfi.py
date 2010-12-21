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
    # type 1: charged hadron - essentially tracking resolutions
    PF_EtResType1 = cms.vdouble(0.05,0,0),
    PF_PhiResType1 = cms.vdouble(0.002),  

    # type 2: EM with track (electron) - essentially tracking resolution
    PF_EtResType2 = cms.vdouble(0.05,0,0),
    PF_PhiResType2 = cms.vdouble(0.002),

    #type 3: muon
    PF_EtResType3 = cms.vdouble(0.05,0,0),
    PF_PhiResType3 = cms.vdouble(0.002),

    # type 4: EM witout track (photon)
    PF_EtResType4 = cms.vdouble(0.2,0.03,0.005),
    PF_PhiResType4 = cms.vdouble(0.005),  

    # type 5: hadron without track (all calorimeter)
    PF_EtResType5 = cms.vdouble(0.,1.22,0.05),
    PF_PhiResType5 = cms.vdouble(0.02511),

    # type 6: hadron without track (Forward HCAL)
    PF_EtResType6 = cms.vdouble(0.,1.22,0.05),
    PF_PhiResType6 = cms.vdouble(0.02511),

    # type 7: EM without track (Forward HCAL)
    PF_EtResType7 = cms.vdouble(0.,1.22,0.05),
    PF_PhiResType7 = cms.vdouble(0.02511)  
    
    )
