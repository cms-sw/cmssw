import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
trackProbabilityFrontierCond = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        messageLevel = cms.untracked.int32(0),
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BTagTrackProbability2DRcd'),
        tag = cms.string('probBTagPDF2D_tag_mc')
    ), 
        cms.PSet(
            record = cms.string('BTagTrackProbability3DRcd'),
            tag = cms.string('probBTagPDF3D_tag_mc')
        )),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_BTAU') ##cms_orcoff_int2r/CMS_COND_BTAU"

)


