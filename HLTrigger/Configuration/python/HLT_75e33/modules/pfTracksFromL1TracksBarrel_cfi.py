import FWCore.ParameterSet.Config as cms

l1tPFTracksFromL1TracksBarrel = cms.EDProducer("PFTrackProducerFromL1Tracks",
    L1TrackTag = cms.InputTag("l1tTTTracksFromTrackletEmulation","Level1TTTracks"),
    nParam = cms.uint32(4),
    resolCalo = cms.PSet(
        etaBins = cms.vdouble(0.7, 1.2, 1.6),
        kind = cms.string('calo'),
        offset = cms.vdouble(2.909, 2.864, 0.294),
        scale = cms.vdouble(0.119, 0.127, 0.442)
    ),
    resolTrack = cms.PSet(
        etaBins = cms.vdouble(0.8, 1.2, 1.5, 2.0, 2.5),
        kind = cms.string('track'),
        offset = cms.vdouble(0.007, 0.009, 0.011, 0.015, 0.025),
        scale = cms.vdouble(0.275, 0.404, 0.512, 0.48, 1.132)
    )
)
