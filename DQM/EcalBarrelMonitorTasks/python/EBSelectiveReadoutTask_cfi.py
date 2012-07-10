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
    configFromCondDB = cms.bool(False),
    # if configFromCondDB is true, dccWeights are not used.
    dccWeights = cms.vdouble(-0.374, -0.374, -0.3629, 0.2721, 0.4681, 0.3707),
    ecalDccZs1stSample = cms.int32(2)
)
