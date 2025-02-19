import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
)

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.testECF = cms.EDAnalyzer("testEcalClusterFunctions",
                #functionName = cms.string( "EcalClusterCrackCorrection" )
                functionName = cms.string( "EcalClusterLocalContCorrection" )
                )

process.p = cms.Path( process.testECF )
