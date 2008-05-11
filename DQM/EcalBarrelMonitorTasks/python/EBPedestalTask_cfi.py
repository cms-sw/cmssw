import FWCore.ParameterSet.Config as cms

ecalBarrelPedestalTask = cms.EDFilter("EBPedestalTask",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    prefixME = cms.untracked.string('EcalBarrel')
)


