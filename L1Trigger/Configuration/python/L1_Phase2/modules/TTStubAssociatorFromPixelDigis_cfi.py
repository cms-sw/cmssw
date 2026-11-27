import FWCore.ParameterSet.Config as cms

TTStubAssociatorFromPixelDigis = cms.EDProducer("TTStubAssociator_Phase2TrackerDigi_",
    TTClusterTruth = cms.VInputTag(cms.InputTag("TTClusterAssociatorFromPixelDigis","ClusterAccepted"), cms.InputTag("TTClusterAssociatorFromPixelDigis","ClusterRejected")),
    TTStubs = cms.VInputTag(cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"), cms.InputTag("TTStubsFromPhase2TrackerDigis","StubRejected"))
)
