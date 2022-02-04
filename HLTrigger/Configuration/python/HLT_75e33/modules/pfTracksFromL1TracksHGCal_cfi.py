import FWCore.ParameterSet.Config as cms

pfTracksFromL1TracksHGCal = cms.EDProducer("PFTrackProducerFromL1Tracks",
    L1TrackTag = cms.InputTag("TTTracksFromTrackletEmulation","Level1TTTracks"),
    nParam = cms.uint32(4),
    resolCalo = cms.PSet(
        etaBins = cms.vdouble(
            1.7, 1.9, 2.2, 2.5, 2.8,
            2.9
        ),
        kind = cms.string('calo'),
        offset = cms.vdouble(
            1.793, 1.827, 2.363, 2.538, 2.812,
            2.642
        ),
        scale = cms.vdouble(
            0.138, 0.137, 0.124, 0.115, 0.106,
            0.121
        )
    ),
    resolTrack = cms.PSet(
        etaBins = cms.vdouble(0.8, 1.2, 1.5, 2.0, 2.5),
        kind = cms.string('track'),
        offset = cms.vdouble(0.007, 0.009, 0.011, 0.015, 0.025),
        scale = cms.vdouble(0.275, 0.404, 0.512, 0.48, 1.132)
    )
)
