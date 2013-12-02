import FWCore.ParameterSet.Config as cms

TTTracksFromPixelDigis = cms.EDProducer("TTTrackBuilder_PixelDigi_",
    TTStubsBricks = cms.InputTag("TTStubsFromPixelDigis", "StubsPass"),
    AssociativeMemories = cms.bool(False)
)



