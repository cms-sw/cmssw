import FWCore.ParameterSet.Config as cms

hcalRawData = cms.EDProducer("HcalDigiToRaw",
    HBHE = cms.untracked.InputTag("simHcalDigis"),
    HF = cms.untracked.InputTag("simHcalDigis"),
    HO = cms.untracked.InputTag("simHcalDigis"),
    ZDC = cms.untracked.InputTag("simHcalUnsuppressedDigis"),
    TRIG =  cms.untracked.InputTag("simHcalTriggerPrimitiveDigis")
)



