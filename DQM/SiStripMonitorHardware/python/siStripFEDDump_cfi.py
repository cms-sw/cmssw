import FWCore.ParameterSet.Config as cms

siStripFEDDump = cms.EDAnalyzer("SiStripFEDDumpPlugin",
  #Raw data collection
  RawDataTag = cms.untracked.InputTag('source'),
  #FED ID to dump
  FEDID = cms.untracked.uint32(50)
)

