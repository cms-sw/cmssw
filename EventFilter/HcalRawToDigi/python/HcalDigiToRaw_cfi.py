import FWCore.ParameterSet.Config as cms

hcalRawData = cms.EDFilter("HcalDigiToRaw",
    HBHE = cms.untracked.InputTag("hcalDigis"),
    HF = cms.untracked.InputTag("hcalDigis"),
    HO = cms.untracked.InputTag("hcalDigis")
)


