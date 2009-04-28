import FWCore.ParameterSet.Config as cms

ecalEndcapSelectiveReadoutTask = cms.EDAnalyzer("EESelectiveReadoutTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    EEDigiCollection = cms.InputTag("ecalEBunpacker:eeDigis"),
    EEUsuppressedDigiCollection = cms.InputTag("ecalUnsuppressedDigis"),
    EESRFlagCollection = cms.InputTag("ecalEBunpacker"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker:EcalTriggerPrimitives"),
    FEDRawDataCollection = cms.InputTag("source"),
    dccWeights = cms.vdouble(-1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266),
    ecalDccZs1stSample = cms.int32(3)

)
