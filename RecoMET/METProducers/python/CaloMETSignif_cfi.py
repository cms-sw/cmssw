import FWCore.ParameterSet.Config as cms
# File: CaloMETSignif_cfi.py
# Author: F.Blekman
# Date: 07.30.2008
#
# Run the missing ET significance algorithm.

# product
metsignificance = cms.EDProducer("METProducer",
    src = cms.InputTag("towerMaker"),
    METType = cms.string('CaloMETSignif'),
    alias = cms.string('RawCaloMETSignif'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection'),
      
    #missing ET significance resolution parameters: 
    # ET resolution has three parameters, defined as parameters in 'standard' resolution function res= et*sqrt((par[2]*par[2])+(par[1]*par[1]/et)+(par[0]*par[0]/(et*et)));
    # PHI resolution has one parameter,defined as res = par[0]*et;

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


