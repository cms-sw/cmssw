import FWCore.ParameterSet.Config as cms

TTTrackAssociatorFromPixelDigisExtended = cms.EDProducer("TTTrackAssociator_Phase2TrackerDigi_",
    TTClusterTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis","ClusterAccepted"),
    TTStubTruth = cms.InputTag("TTStubAssociatorFromPixelDigis","StubAccepted"),
    TTTrackAllowOneFalse2SStub = cms.bool(True),
    TTTracks = cms.VInputTag(cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation","Level1TTTracks"))
)
