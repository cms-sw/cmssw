import FWCore.ParameterSet.Config as cms

me0TriggerConvertedPseudoDigis = cms.EDProducer("ME0TriggerPseudoProducer",
    ME0SegmentProducer = cms.InputTag("me0Segments"),
    info = cms.untracked.int32(0),
    DeltaPhiResolution = cms.untracked.double(0.25)# in term of trigger pad
)
