import FWCore.ParameterSet.Config as cms


hcalRawDataVME = cms.EDProducer("HcalDigiToRaw",
    HBHE = cms.untracked.InputTag("simHcalDigis"),
    HF = cms.untracked.InputTag("simHcalDigis"),
    HO = cms.untracked.InputTag("simHcalDigis"),
    ZDC = cms.untracked.InputTag("simHcalUnsuppressedDigis"),
    TRIG =  cms.untracked.InputTag("simHcalTriggerPrimitiveDigis")
)

hcalRawData = cms.Sequence(hcalRawDataVME)

from EventFilter.HcalRawToDigi.hcalDigiToRawuHTR_cfi import hcalDigiToRawuHTR as hcalRawDatauHTR

_phase1_hcalRawData = hcalRawData.copy()
_phase1_hcalRawData += hcalRawDatauHTR

from Configuration.StandardSequences.Eras import eras
eras.run2_HCAL_2017.toModify( hcalRawDataVME,
    HBHE = cms.untracked.InputTag(""),
    HF = cms.untracked.InputTag(""),
    TRIG = cms.untracked.InputTag("")
)
eras.run2_HCAL_2017.toReplaceWith(hcalRawData,_phase1_hcalRawData)
