import FWCore.ParameterSet.Config as cms

# Reading from DB
from CondCore.DBCommon.CondDBSetup_cfi import *
PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TibTidTecAllSurvey_v2')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorExtendedRcd'),
            tag = cms.string('TibTidTecAllSurveyAPE_v2')
        ))
)

