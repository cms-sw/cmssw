import FWCore.ParameterSet.Config as cms

hcalRawData = cms.EDFilter("HcalDigiToRaw",
    HBHE = cms.untracked.InputTag("simHcalDigis"),
    HF = cms.untracked.InputTag("simHcalDigis"),
    HO = cms.untracked.InputTag("simHcalDigis")
)



