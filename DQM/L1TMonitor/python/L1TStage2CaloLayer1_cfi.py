import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer1 = DQMStep1Module('L1TStage2CaloLayer1',
    ecalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    hcalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    ecalTPSourceSent = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    hcalTPSourceSent = cms.InputTag("hcalDigis"),
    fedRawDataLabel  = cms.InputTag("rawDataCollector"),
    histFolder = cms.string('L1T/L1TStage2CaloLayer1'),
)
