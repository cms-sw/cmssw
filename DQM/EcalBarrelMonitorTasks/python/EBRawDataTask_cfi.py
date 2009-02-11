import FWCore.ParameterSet.Config as cms

ecalBarrelRawDataTask = cms.EDFilter("EBRawDataTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    FEDRawDataCollection = cms.InputTag("source"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    GTEvmSource = cms.InputTag("l1GtEvmUnpack")
)
