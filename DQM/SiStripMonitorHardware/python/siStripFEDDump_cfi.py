import FWCore.ParameterSet.Config as cms

siStripFEDDump = DQMStep1Module('SiStripFEDDumpPlugin',
  #Raw data collection
  RawDataTag = cms.untracked.InputTag('source'),
  #FED ID to dump
  FEDID = cms.untracked.uint32(50)
)

