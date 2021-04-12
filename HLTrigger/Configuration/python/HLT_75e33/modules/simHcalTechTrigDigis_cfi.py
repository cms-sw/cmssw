import FWCore.ParameterSet.Config as cms

simHcalTechTrigDigis = cms.EDProducer("HcalTTPTriggerRecord",
    ttpBitNames = cms.vstring(
        'L1Tech_HCAL_HF_MM_or_PP_or_PM.v0',
        'L1Tech_HCAL_HF_coincidence_PM.v1',
        'L1Tech_HCAL_HF_MMP_or_MPP.v0'
    ),
    ttpBits = cms.vuint32(8, 9, 10),
    ttpDigiCollection = cms.InputTag("simHcalTTPDigis")
)
