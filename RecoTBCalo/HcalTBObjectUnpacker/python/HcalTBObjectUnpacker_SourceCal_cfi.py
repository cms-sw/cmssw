import FWCore.ParameterSet.Config as cms

tbunpack = cms.EDFilter("HcalTBObjectUnpacker",
    HcalSlowDataFED = cms.untracked.int32(-1),
    HcalSourcePositionFED = cms.untracked.int32(6),
    HcalTriggerFED = cms.untracked.int32(1)
)


