import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:PhysicsPerformance.db'

process.load ("RecoBTag.PerformanceDB.BtagPerformanceFragment")
process.load ("RecoBTag.PerformanceDB.PoolBtagEffFragment")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.demo2 = cms.EDAnalyzer('TestPerformanceFW_ES',
                       AlgoName = cms.string('JetProbability_loose'))



#
# change inside the analyzer
#

process.demo2.AlgoName = "JetProbability_medium"

process.p = cms.Path(process.demo2)

#

