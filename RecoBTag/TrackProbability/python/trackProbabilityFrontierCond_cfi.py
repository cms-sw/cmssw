import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
trackProbabilityFrontierCond = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        connectionRetrialPeriod = cms.untracked.int32(30),
        loadBlobStreamer = cms.untracked.bool(False),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(True),
        enableConnectionSharing = cms.untracked.bool(False),
        connectionRetrialTimeOut = cms.untracked.int32(180),
        connectionTimeOut = cms.untracked.int32(600),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    timetype = cms.string('runnumber'),
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


