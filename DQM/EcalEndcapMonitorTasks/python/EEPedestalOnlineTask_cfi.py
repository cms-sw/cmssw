import FWCore.ParameterSet.Config as cms

ecalEndcapPedestalOnlineTask = cms.EDAnalyzer("EEPedestalOnlineTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    subfolder = cms.untracked.string(""),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")
)

