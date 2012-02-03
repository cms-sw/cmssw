import FWCore.ParameterSet.Config as cms

l1TdeDTTPG = cms.EDAnalyzer("L1TdeDTTPG",
     dataTag     = cms.untracked.InputTag("dttfDigis"),
     emulatorTag = cms.untracked.InputTag("valDtTriggerPrimitiveDigis"),
     gmtTag      = cms.untracked.InputTag("gtDigis"),
     detailedAnalysis = cms.untracked.bool(True),
     ResetCycle       = cms.untracked.int32(9999)
)


