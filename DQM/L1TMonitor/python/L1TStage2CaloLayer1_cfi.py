import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer1 = cms.EDAnalyzer("L1TStage2CaloLayer1",
    ecalTPSourceRecd = cms.InputTag("l1tCaloLayer1Digis"),
    hcalTPSourceRecd = cms.InputTag("l1tCaloLayer1Digis"),
    ecalTPSourceSent = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    hcalTPSourceSent = cms.InputTag("hcalDigis"),
    fedRawDataLabel  = cms.InputTag("rawDataCollector"),
    histFolder = cms.string('L1T/L1TStage2CaloLayer1'),
    # HF minBias bit is not yet unpacked on HCAL side, so we don't compare them
    ignoreHFfb2 = cms.untracked.bool(True),
)
