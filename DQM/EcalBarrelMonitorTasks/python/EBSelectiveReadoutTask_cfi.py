import FWCore.ParameterSet.Config as cms

ecalBarrelSelectiveReadoutTask = cms.EDFilter("EBSelectiveReadoutTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    EBDigiCollection = cms.InputTag("ecalEBunpacker:ebDigis"),
    EBUsuppressedDigiCollection = cms.InputTag("ecalUnsuppressedDigis"),
    EBSRFlagCollection = cms.InputTag("ecalEBunpacker"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker:EcalTriggerPrimitives"),
    EcalFEDRawCollection = cms.InputTag("source")
)
