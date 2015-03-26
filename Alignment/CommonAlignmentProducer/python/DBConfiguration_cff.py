import FWCore.ParameterSet.Config as cms

#
# Database output service
# Required if AlignmentProducer.saveToDB = true
from CondCore.DBCommon.CondDBSetup_cfi import *
PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    #string connect = "frontier://FrontierProd/CMS_COND_20X_ALIGNMENT"
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry200')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorExtendedRcd'),
            tag = cms.string('TrackerIdealGeometryErrors200')
        )),
    connect = cms.string('sqlite_file:Alignments.db')
)

PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dataout.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('<output tag>')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorExtendedRcd'),
            tag = cms.string('<output error tag>')
        ))
)


