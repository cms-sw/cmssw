# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTHVStatusRcd'),
        tag = cms.string('hv_test')
    )),
    connect = cms.string('sqlite_file:testhv.db'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    )
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('timestamp'),
    firstValue = cms.uint64(5412014424843000000),
    lastValue  = cms.uint64(5412014424843000000),
#    firstValue = cms.uint64(5374055233297548744),
#    lastValue  = cms.uint64(5374055233297548744),
    interval   = cms.uint64(1)
)

#process.source = cms.Source("EmptySource",
#    numberEventsInRun = cms.untracked.uint32(1),
#    firstRun = cms.untracked.uint32(54544)
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.hv = cms.EDAnalyzer("DTHVDump")

process.p = cms.Path(process.hv)

