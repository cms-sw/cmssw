import FWCore.ParameterSet.Config as cms

ecalEndcapStatusFlagsTask = cms.EDFilter("EEStatusFlagsTask",
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    prefixME = cms.untracked.string('EcalEndcap')
)


