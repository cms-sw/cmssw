import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.CSCFakeGainsConditions = cms.ESSource("CSCFakeGainsConditions")

process.prefer("CSCFakeGainsConditions")
process.CSCFakePedestalsConditions = cms.ESSource("CSCFakePedestalsConditions")

process.prefer("CSCFakePedestalsConditions")
process.CSCFakeNoiseMatrixConditions = cms.ESSource("CSCFakeNoiseMatrixConditions")

process.prefer("CSCFakeNoiseMatrixConditions")
process.CSCFakeCrosstalkConditions = cms.ESSource("CSCFakeCrosstalkConditions")

process.prefer("CSCFakeCrosstalkConditions")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod1 = cms.EDAnalyzer("CSCGainsReadAnalyzer")

process.prod2 = cms.EDAnalyzer("CSCPedestalReadAnalyzer")

process.prod3 = cms.EDAnalyzer("CSCCrossTalkReadAnalyzer")

process.prod4 = cms.EDAnalyzer("CSCNoiseMatrixReadAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod1*process.prod2*process.prod3*process.prod4)
process.ep = cms.EndPath(process.output)

