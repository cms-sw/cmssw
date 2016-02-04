import FWCore.ParameterSet.Config as cms

rctSaveInput = cms.EDAnalyzer("L1RCTSaveInput",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    useDebugTpgScales = cms.bool(False),
    rctTestInputFile = cms.untracked.string('rctSaveTest.txt'),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis")
)



