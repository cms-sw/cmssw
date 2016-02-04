import FWCore.ParameterSet.Config as cms

ecalBarrelSelectiveReadoutTask = cms.EDAnalyzer("EBSelectiveReadoutTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    EBDigiCollection = cms.InputTag("ecalEBunpacker:ebDigis"),
    EBUsuppressedDigiCollection = cms.InputTag("ecalUnsuppressedDigis"),
    EBSRFlagCollection = cms.InputTag("ecalEBunpacker"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker:EcalTriggerPrimitives"),
    FEDRawDataCollection = cms.InputTag("source"),
    dccWeights = cms.vdouble(-1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266),
    ecalDccZs1stSample = cms.int32(3)

)
