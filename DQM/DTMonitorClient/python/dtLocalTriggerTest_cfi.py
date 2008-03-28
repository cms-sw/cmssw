import FWCore.ParameterSet.Config as cms

triggerTest = cms.EDFilter("DTLocalTriggerTest",
    runningStandalone = cms.untracked.bool(True),
    diagnosticPrescale = cms.untracked.int32(1),
    folderRoot = cms.untracked.string(''),
    dataFromDDU = cms.untracked.bool(True)
)


