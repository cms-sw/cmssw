import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

#Load up our measurements!
#Data measurements from Spring11
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDB1107")
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDB1107")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

#Setup the analyzer.
#process.demo2 = cms.EDAnalyzer('DumpBtagTable',
#                               AlgoName = cms.string('JetProbability_loose'),
#                               measureName = cms.vstring("MISTAGSSVHEM","MISTAGSSVHEM","MISTAGSSVHEM","MISTAGSSVHEM",
#                                                         "MISTAGSSVHPT","MISTAGSSVHPT","MISTAGSSVHPT","MISTAGSSVHPT",
#                                                         "MISTAGTCHEM","MISTAGTCHEM","MISTAGTCHEM","MISTAGTCHEM"),
#                              measureType = cms.vstring("BTAGLEFF", "BTAGLERR", "BTAGLEFFCORR", "BTAGLERRCORR",
#                                                         "BTAGLEFF", "BTAGLERR", "BTAGLEFFCORR", "BTAGLERRCORR",
#                                                         "BTAGLEFF", "BTAGLERR", "BTAGLEFFCORR", "BTAGLERRCORR")
#                               )

## DUMP Tables for CSVM: data b-tag efficiencies, SF_b, data mistag efficiencies, SF_mistag  
process.demo2 = cms.EDAnalyzer('DumpBtagTable',
                               AlgoName = cms.string('CombinedSecondaryVertex_medium'),
                               measureName = cms.vstring("BTAGCSVM","BTAGCSVM","MISTAGCSVM","MISTAGCSVM"),
                               measureType = cms.vstring("BTAGBEFF", "BTAGBEFFCORR", "BTAGLEFF", "BTAGLEFFCORR")
                               )

#process.p = cms.Path(process.demo2 * process.demo3)
process.p = cms.Path(process.demo2)


