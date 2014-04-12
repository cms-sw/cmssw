import FWCore.ParameterSet.Config as cms

fakeAlignment = cms.ESProducer("FakeAlignmentProducer",
    appendToDataLabel = cms.string('')
)


