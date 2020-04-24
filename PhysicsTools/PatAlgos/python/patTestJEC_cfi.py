import FWCore.ParameterSet.Config as cms

# Be sure to change the "V5" to whatever is in your payloads. 
from CondCore.DBCommon.CondDBSetup_cfi import *
jec = cms.ESSource("PoolDBESSource",CondDBSetup,
                   connect = cms.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS"),
                   toGet =  cms.VPSet(
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_AK4Calo"),
                                label=cms.untracked.string("AK4Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_AK4PF"),
                                label=cms.untracked.string("AK4PF")),                                   
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_AK4JPT"),
                                label=cms.untracked.string("AK4JPT")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_AK4TRK"),
                                label=cms.untracked.string("AK4TRK")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_AK7Calo"),
                                label=cms.untracked.string("AK7Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_AK7PF"),
                                label=cms.untracked.string("AK7PF")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_IC5Calo"),
                                label=cms.untracked.string("IC5Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_IC5PF"),
                                label=cms.untracked.string("IC5PF")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_KT4Calo"),
                                label=cms.untracked.string("KT4Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_KT4PF"),
                                label=cms.untracked.string("KT4PF")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_KT6Calo"),
                                label=cms.untracked.string("KT6PF")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec10V1_KT6PF"),
                                label=cms.untracked.string("KT6PF")),
                       )
                   
                   )

es_prefer_jec = cms.ESPrefer("PoolDBESSource","jec")
