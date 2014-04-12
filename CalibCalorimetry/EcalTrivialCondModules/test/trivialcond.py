import FWCore.ParameterSet.Config as cms

process = cms.Process("testTrivial")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
)

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.testTrivial = cms.EDAnalyzer("EcalTrivialObjectAnalyzer")

process.p = cms.Path( process.testTrivial )
