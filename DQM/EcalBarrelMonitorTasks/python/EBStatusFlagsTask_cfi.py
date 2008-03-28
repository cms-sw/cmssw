import FWCore.ParameterSet.Config as cms

ecalBarrelStatusFlagsTask = cms.EDFilter("EBStatusFlagsTask",
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(True)
)


