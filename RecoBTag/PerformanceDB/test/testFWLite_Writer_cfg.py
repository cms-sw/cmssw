import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load("CondCore.DBCommon.CondDBCommon_cfi") 
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDBMC36X") 
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDBMC36X")
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDB1011")
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDB1011") 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.TFileService = cms.Service("TFileService", fileName = cms.string("performance_ssvm.root") )


process.myrootwriter = cms.EDAnalyzer("BTagPerformaceRootProducerFromSQLITE",
                                  names = cms.vstring('MCPfTCHEMb', 'MISTAGSSVHEM'),
                                  index = cms.uint32(1001)
                                  )


process.p = cms.Path(process.myrootwriter)
