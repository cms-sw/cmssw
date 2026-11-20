import FWCore.ParameterSet.Config as cms

ge0TriggerConvertedPseudoDigis = cms.EDProducer("GE0TriggerPseudoProducer",
    DeltaPhiResolution = cms.untracked.double(0.25),
    ME0SegmentProducer = cms.InputTag("gemSegments"),
    info = cms.untracked.int32(0)
)
