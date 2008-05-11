import FWCore.ParameterSet.Config as cms

ecalEndcapPedestalTask = cms.EDFilter("EEPedestalTask",
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    mergeRuns = cms.untracked.bool(False),    
    enableCleanup = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalEndcap')
)


