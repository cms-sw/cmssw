import FWCore.ParameterSet.Config as cms

process = cms.Process("ConvertToEDM")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("myConf.sources.source_cff")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

#process.p = cms.Path( process.dump )

process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("edmOutput.root")
)

process.e = cms.EndPath( process.output )
