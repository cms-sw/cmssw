import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load("CondCore.DBCommon.CondDBCommon_cfi") 
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDB062012")
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDB062012")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.TFileService = cms.Service("TFileService", fileName = cms.string("btag_performance_2012.root") )


process.myrootwriter = cms.EDAnalyzer("BTagPerformaceRootProducerFromSQLITE",
                                  names = cms.vstring('TTBARWPBTAGCSVL','TTBARWPBTAGCSVM','TTBARWPBTAGCSVT','TTBARWPBTAGJPL','TTBARWPBTAGJPM','TTBARWPBTAGJPT','MISTAGCSVL','MISTAGCSVM','MISTAGCSVT','MISTAGJBPL','MISTAGJBPM','MISTAGJBPT'),
                                  index = cms.uint32(1001)
                                  )


process.p = cms.Path(process.myrootwriter)
