import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(
                cms.PSet( record = cms.string('DTRangeT0Rcd'),
                          tag = cms.string('rT0_test') )
    ),
    connect = cms.string('sqlite_file:test.db'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    )
)

process.source = cms.Source("EmptySource",
    firstRun  = cms.untracked.uint32(1),
    numberEventsInRun  = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.get = cms.EDAnalyzer("DTRangeT0Print")

process.p = cms.Path(process.get)


