import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTECALUNPACKERDATA")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring('file:/data/franzoni/data/h4b.00013509.A.0.0.root')
)

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")
  
process.p = cms.Path(process.ecalEBunpacker*process.dump)
