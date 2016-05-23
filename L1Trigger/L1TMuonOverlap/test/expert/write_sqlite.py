import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:Patterns.db'

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('L1TMTFOverlapParamsRcd'),
        tag = cms.string('OMTFParams_test')
    ))
)

###OMTF CondFormats ESProducer
process.load('L1Trigger.L1TMuonOverlap.fakeOmtfParams_cff')

###EDAnalyzer writing the OMTFParams objects to sqlite file
process.write_sqlite = cms.EDAnalyzer("L1MuonOverlapParamsDBProducer")

process.p = cms.Path(process.write_sqlite)
