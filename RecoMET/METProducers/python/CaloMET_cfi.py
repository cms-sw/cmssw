import FWCore.ParameterSet.Config as cms

# File: CaloMET.cff
# Original Author: R. Cavanaugh
# Date: 08.08.2006
#
# Form uncorrected Missing ET from Calorimeter Towers and store into event as a CaloMET
# product

# Modification by F. Ratnikov and R. Remington
# Date: 10/21/08
# Additional modules available for MET Reconstruction using towers w/wo HO included


met = cms.EDProducer(
    "METProducer",
    src = cms.InputTag("towerMaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection'),
    
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

metHO = met.clone()
metHO.src = "towerMakerWithHO"
metHO.alias = 'RawCaloMETHO'

metOpt = cms.EDProducer(
    "METProducer",
    src = cms.InputTag("calotoweroptmaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOpt'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection'),

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

metOptHO = metOpt.clone()
metOptHO.src = "calotoweroptmakerWithHO"
metOptHO.alias = 'RawCaloMETOptHO'

metNoHF = cms.EDProducer(
    "METProducer",
    src = cms.InputTag("towerMaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection'),

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

metNoHFHO = metNoHF.clone()
metNoHFHO.src = "towerMakerWithHO"
metNoHFHO.alias = 'RawCaloMETNoHFHO'

metOptNoHF = cms.EDProducer(
    "METProducer",
    src = cms.InputTag("calotoweroptmaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOptNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection'),

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

metOptNoHFHO = metOptNoHF.clone()
metOptNoHFHO.src = "calotoweroptmakerWithHO"
metOptNoHFHO.alias = 'RawCaloMETOptNoHFHO'


