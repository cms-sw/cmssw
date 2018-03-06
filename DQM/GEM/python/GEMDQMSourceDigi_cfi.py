import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

GEMDQMSourceDigi = DQMEDAnalyzer("GEMDQMSourceDigi",
    digisInputLabel = cms.InputTag("muonGEMDigis", ""),
)
