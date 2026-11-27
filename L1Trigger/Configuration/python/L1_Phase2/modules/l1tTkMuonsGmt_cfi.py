import FWCore.ParameterSet.Config as cms

l1tTkMuonsGmt = cms.EDProducer("Phase2L1TGMTTkMuonProducer",
    isolation = cms.PSet(
        AbsIsoThresholdL = cms.int32(160),
        AbsIsoThresholdM = cms.int32(120),
        AbsIsoThresholdT = cms.int32(80),
        IsodumpForHLS = cms.int32(0),
        RelIsoThresholdL = cms.double(0.1),
        RelIsoThresholdM = cms.double(0.05),
        RelIsoThresholdT = cms.double(0.01),
        verbose = cms.int32(0)
    ),
    minTrackStubs = cms.int32(4),
    muonBXMax = cms.int32(0),
    muonBXMin = cms.int32(0),
    srcStubs = cms.InputTag("l1tStubsGmt","tps"),
    srcTracks = cms.InputTag("l1tTTTracksFromTrackletEmulation","Level1TTTracks"),
    tauto3mu = cms.PSet(

    ),
    trackConverter = cms.PSet(
        verbose = cms.int32(0)
    ),
    trackMatching = cms.PSet(
        verbose = cms.int32(0)
    ),
    verbose = cms.int32(0)
)
