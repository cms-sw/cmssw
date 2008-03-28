import FWCore.ParameterSet.Config as cms

# Database output service
# Required if AlignmentProducer.saveToDB = true
from CondCore.DBCommon.CondDBSetup_cfi import *
PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Tracker10pbScenario')
    ), cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('Tracker10pbScenarioErrors')
    )),
    connect = cms.string('sqlite_file:Alignments.db')
)

PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:dataout.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('<output tag>')
    ), cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('<output error tag>')
    ))
)


