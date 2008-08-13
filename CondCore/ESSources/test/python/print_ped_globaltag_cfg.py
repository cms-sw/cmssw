import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    connect = cms.string('sqlite_file:tagDB.db'),
    globaltag = cms.string('Calibration')
)

process.source = cms.Source("EmptyIOVSource",
    lastRun = cms.untracked.uint32(3),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

process.prod = cms.EDAnalyzer("PedestalsByLabelAnalyzer")

process.p = cms.Path(process.prod)


