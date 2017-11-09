import FWCore.ParameterSet.Config as cms

GEMDQMSourceDigi = cms.EDAnalyzer("GEMDQMSourceDigi",
    digisInputLabel = cms.InputTag("muonGEMDigis", "", "RECO"),
    errorsInputLabel = cms.InputTag("muonGEMDigis", "vfatErrors", "RECO")     
  
)
