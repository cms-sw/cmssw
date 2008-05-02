import FWCore.ParameterSet.Config as cms

ecalEndcapPedestalTask = cms.EDFilter("EEPedestalTask",
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False)
)


