import FWCore.ParameterSet.Config as cms

ecalBarrelPedestalOnlineTask = cms.EDAnalyzer("EBPedestalOnlineTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    subfolder = cms.untracked.string(""),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EBDigiCollection = cms.InputTag("ecalDigis","ebDigis")
)

