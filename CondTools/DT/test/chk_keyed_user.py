# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('DTCCBConfigRcd'),
    tag = cms.string('conf_test')
    ),
    cms.PSet(
    record = cms.string('DTKeyedConfigListRcd'),
    tag = cms.string('keyedConfListIOV_V01')
    ),
    cms.PSet(
    record = cms.string('DTKeyedConfigContainerRcd'),
    tag = cms.string('keyedConfBricks_V01')
    )
    ),
    connect = cms.string('sqlite_file:userconf.db'),
#    connect = cms.string('sqlite_file:testconf.db'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    )
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(54544)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.conf = cms.EDAnalyzer("DTKeyedConfigDump",
    dumpCCBKeys = cms.bool(True),
#    dumpCCBKeys = cms.bool(False),
    dumpAllData = cms.bool(True)
#    dumpAllData = cms.bool(False)
)

process.p = cms.Path(process.conf)

