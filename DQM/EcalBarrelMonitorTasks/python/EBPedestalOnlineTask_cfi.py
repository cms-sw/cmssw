import FWCore.ParameterSet.Config as cms

ecalBarrelPedestalOnlineTask = cms.EDAnalyzer("EBPedestalOnlineTask",
    prefixME = cms.untracked.string('Ecal'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis")
)

