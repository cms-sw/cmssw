import FWCore.ParameterSet.Config as cms

# Be sure to change the "V5" to whatever is in your payloads. 
from CondCore.DBCommon.CondDBSetup_cfi import *
jec = cms.ESSource("PoolDBESSource",CondDBSetup,
                   connect = cms.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS"),
                   toGet =  cms.VPSet(
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Spring10_V8_AK5Calo"),
                                label=cms.untracked.string("AK5Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Spring10_V8_AK5PF"),
                                label=cms.untracked.string("AK5PF")),                                   
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Summer10_V8_AK5JPT"),
                                label=cms.untracked.string("AK5JPT")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Spring10_V8_IC5Calo"),
                                label=cms.untracked.string("IC5Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Spring10_V8_KT6Calo"),
                                label=cms.untracked.string("KT6Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Spring10_V8_KT4Calo"),
                                label=cms.untracked.string("KT4Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Spring10_V8_AK7Calo"),
                                label=cms.untracked.string("AK7Calo")),
                       )
                   
                   )
es_prefer_jec = cms.ESPrefer("PoolDBESSource","jec")

