import FWCore.ParameterSet.Config as cms

hcalRawDatauHTR = cms.EDProducer("HcalDigiToRawuHTR",
    ElectronicsMap = cms.string("")
)

