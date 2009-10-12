import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:PhysicsPerformance.db'

process.load ("RecoBTag.PerformanceDB.BtagPerformanceESProducer_cfi")
#
# change inside the source
#

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBCommon,
                                      toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('TEST_TABLE_1_WP'),
    label = cms.untracked.string('TEST_TABLE_1_WP')
    ),
                                                        cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('TEST_TABLE_1'),
    label = cms.untracked.string('TEST_TABLE_1')
    ))

)

process.demo2 = cms.EDAnalyzer('TestPerformanceFW_ES',
                       AlgoName = cms.string('JetProbability_loose'))

#
# change inside the source
#
process.BtagPerformanceESProducer_test.PayloadName = "TEST_TABLE_1"
process.BtagPerformanceESProducer_test.WorkingPointName = "TEST_TABLE_1_WP"

process.p = cms.Path(process.demo2)

#

