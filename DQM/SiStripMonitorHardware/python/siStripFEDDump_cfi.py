import FWCore.ParameterSet.Config as cms

siStripFEDDump = cms.EDAnalyzer("SiStripFEDDumpPlugin",
  #Raw data collection
  RawDataTag = cms.InputTag('source'),
  #FED ID to dump
  FEDID = cms.uint32(50)
)

