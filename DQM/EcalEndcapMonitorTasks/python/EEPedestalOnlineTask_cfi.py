import FWCore.ParameterSet.Config as cms

ecalEndcapPedestalOnlineTask = cms.EDFilter("EEPedestalOnlineTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")
)

