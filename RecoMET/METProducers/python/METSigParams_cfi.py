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
    HF_PhiResPar = cms.vdouble(0.05022)     # 0.174/sqrt(12)
    
    )
