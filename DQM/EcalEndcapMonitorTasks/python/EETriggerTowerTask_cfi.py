import FWCore.ParameterSet.Config as cms

ecalEndcapTriggerTowerTask = cms.EDFilter("EETriggerTowerTask",
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    OutputRootFile = cms.untracked.string(''),
    enableCleanup = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalEndcap'),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("ecalTriggerPrimitiveDigis")
)


