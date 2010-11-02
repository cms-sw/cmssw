import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

#Load up our measurements!
#MC measurements from 36X
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDBMC36X")
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDBMC36X")
#Data measurements from Fall10
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDB1011")
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDB1011")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

#Setup the analyzer.
process.demo2 = cms.EDAnalyzer('TestPerformanceFW_ES',
                               AlgoName = cms.string('JetProbability_loose'),
                               measureName = cms.vstring("MISTAGSSVHEM","MISTAGSSVHEM","MISTAGSSVHEM","MISTAGSSVHEM",
                                                         "MISTAGSSVHPT","MISTAGSSVHPT","MISTAGSSVHPT","MISTAGSSVHPT",
                                                         "MISTAGTCHEM","MISTAGTCHEM","MISTAGTCHEM","MISTAGTCHEM"),
                               measureType = cms.vstring("BTAGLEFF", "BTAGLERR", "BTAGLEFFCORR", "BTAGLERRCORR",
                                                         "BTAGLEFF", "BTAGLERR", "BTAGLEFFCORR", "BTAGLERRCORR",
                                                         "BTAGLEFF", "BTAGLERR", "BTAGLEFFCORR", "BTAGLERRCORR")
                               )

process.p = cms.Path(process.demo2)


