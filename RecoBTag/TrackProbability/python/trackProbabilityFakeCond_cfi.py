import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
trackProbabilityFakeCond = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('.'),
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
    connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/btagTrackProbability200.db')
)


