import FWCore.ParameterSet.Config as cms

tbunpack = cms.EDProducer("HcalTBObjectUnpacker",
    IncludeUnmatchedHits = cms.untracked.bool(False),
    HcalTDCFED = cms.untracked.int32(5),
    HcalSlowDataFED = cms.untracked.int32(3),
    HcalTriggerFED = cms.untracked.int32(1)
)


