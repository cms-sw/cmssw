# The following comments couldn't be translated into the new config version:

# FRONTIER

import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
trackerAlignment = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentRcd'),
            tag = cms.string('TrackerIdealGeometry200')
        ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            tag = cms.string('TrackerIdealGeometryErrors200')
        )),
    connect = cms.string('frontier://FrontierProd/CMS_COND_20X_ALIGNMENT')
)

es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")

