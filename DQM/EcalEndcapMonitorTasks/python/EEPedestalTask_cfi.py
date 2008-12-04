import FWCore.ParameterSet.Config as cms

ecalEndcapPedestalTask = cms.EDFilter("EEPedestalTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    mergeRuns = cms.untracked.bool(False),    
    enableCleanup = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker")
)

