import FWCore.ParameterSet.Config as cms

ProducerDTC = cms.EDProducer("trackerDTC::ProducerDTC",
    BranchAccepted = cms.string('StubAccepted'),
    BranchLost = cms.string('StubLost'),
    InputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted")
)
