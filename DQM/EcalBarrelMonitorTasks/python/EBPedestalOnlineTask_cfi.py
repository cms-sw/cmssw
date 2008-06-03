import FWCore.ParameterSet.Config as cms

ecalBarrelPedestalOnlineTask = cms.EDFilter("EBPedestalOnlineTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis")
)

