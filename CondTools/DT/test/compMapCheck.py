import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
                cms.PSet( record = cms.string('DTReadOutMappingRcd'),
                          tag = cms.string('compactmap') ),
    ),
###    connect = cms.string('sqlite_file:testmap.db'),
    connect = cms.string('sqlite_file:testMCmap.db'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval = cms.uint64(1)
)

process.map   = cms.EDAnalyzer("DTCompMapValidateDBRead",
###    chkFile = cms.string('expmap.txt'),
###    chkFile = cms.string('cruzetmap.txt'),
###    chkFile = cms.string('mcMap/DTRO_770_774.txt'),
    chkFile = cms.string('mcMap/DTROMap.txt'),
    logFile = cms.string('mapValidate.log')
)

process.p = cms.Path(process.map)
