import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

#process.CondDBCommon.connect = 'sqlite_file:PhysicsPerformance.db'
#process.CondDBCommon.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

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
            record = cms.string('PerformancePayloadRecord'),
            tag = cms.string('BTagTTBARDISCRIMBTAGCSVtable'),
            label = cms.untracked.string('BTagTTBARDISCRIMBTAGCSVtable')
            ),
        cms.PSet(
            record = cms.string('PerformanceWPRecord'),
            tag = cms.string('BTagTTBARDISCRIMBTAGCSVwp'),
            label = cms.untracked.string('BTagTTBARDISCRIMBTAGCSVwp')
            )                             


                                                        )
)

process.PoolDBESSource.connect = 'frontier://FrontierProd/CMS_COND_PAT_000'

process.demo2 = cms.EDAnalyzer('TestPerformanceFW_ES_TFormula',
                       AlgoName = cms.string('TTBARDISCRIMBTAGCSV'))
#
# change inside the source
#
#process.BtagPerformanceESProducer_test.PayloadName = "TEST_TF_1"
#process.BtagPerformanceESProducer_test.WorkingPointName = "TEST_TF_1_WP"

process.BtagPerformanceESProducer_test.ComponentName = "TTBARDISCRIMBTAGCSV"
process.BtagPerformanceESProducer_test.PayloadName = "BTagTTBARDISCRIMBTAGCSVtable"
process.BtagPerformanceESProducer_test.WorkingPointName = "BTagTTBARDISCRIMBTAGCSVwp"
                                                           

process.p = cms.Path(process.demo2)

#

# foo bar baz
# 8N7TPPm9tG5fo
# jjIohqdg5qXOS
