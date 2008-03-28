import FWCore.ParameterSet.Config as cms

ecalEndcapPedestalTask = cms.EDFilter("EEPedestalTask",
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(True)
)


