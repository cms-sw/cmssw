import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
siStripFEDDump = DQMEDAnalyzer('SiStripFEDDumpPlugin',
  #Raw data collection
  RawDataTag = cms.untracked.InputTag('source'),
  #FED ID to dump
  FEDID = cms.untracked.uint32(50)
)

