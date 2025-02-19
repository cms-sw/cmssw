import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.CSCFakeDBGains = cms.ESSource("CSCFakeDBGains")

process.prefer("CSCFakeDBGains")
process.CSCFakeDBPedestals = cms.ESSource("CSCFakeDBPedestals")

process.prefer("CSCFakeDBPedestals")
process.CSCFakeDBNoiseMatrix = cms.ESSource("CSCFakeDBNoiseMatrix")

process.prefer("CSCFakeDBNoiseMatrix")
process.CSCFakeDBCrosstalk = cms.ESSource("CSCFakeDBCrosstalk")

process.prefer("CSCFakeDBCrosstalk")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod1 = cms.EDAnalyzer("CSCGainsDBReadAnalyzer")

process.prod2 = cms.EDAnalyzer("CSCPedestalDBReadAnalyzer")

process.prod3 = cms.EDAnalyzer("CSCCrossTalkDBReadAnalyzer")

process.prod4 = cms.EDAnalyzer("CSCNoiseMatrixDBReadAnalyzer")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod1*process.prod2*process.prod3*process.prod4)
process.ep = cms.EndPath(process.output)

