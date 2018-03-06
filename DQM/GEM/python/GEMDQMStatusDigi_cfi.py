import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

GEMDQMStatusDigi = DQMEDAnalyzer("GEMDQMStatusDigi",
    VFATInputLabel = cms.InputTag("muonGEMDigis", "vfatStatus"),
    AMCInputLabel = cms.InputTag("muonGEMDigis", "AMCStatus"),     
    GEBInputLabel = cms.InputTag("muonGEMDigis", "GEBStatus"), 
)
