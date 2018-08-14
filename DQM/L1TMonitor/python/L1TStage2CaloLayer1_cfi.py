import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2CaloLayer1 = DQMEDAnalyzer('L1TStage2CaloLayer1',
    ecalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    hcalTPSourceRecd = cms.InputTag("caloLayer1Digis"),
    ecalTPSourceSent = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    hcalTPSourceSent = cms.InputTag("hcalDigis"),
    fedRawDataLabel  = cms.InputTag("rawDataCollector"),
    histFolder = cms.string('L1T/L1TStage2CaloLayer1'),
    ignoreHFfb2 = cms.untracked.bool(False),
)
