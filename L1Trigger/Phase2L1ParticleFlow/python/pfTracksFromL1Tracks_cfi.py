import FWCore.ParameterSet.Config as cms

pfTracksFromL1Tracks = cms.EDProducer("PFTrackProducerFromL1Tracks",
    L1TrackTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
    nParam = cms.uint32(4),
    resolCalo = cms.PSet(
            etaBins = cms.vdouble( 1.300,  1.700,  2.800,  3.200,  4.000,  5.000),
            offset  = cms.vdouble( 2.644,  1.975,  2.287,  1.113,  0.772,  0.232),
            scale   = cms.vdouble( 0.155,  0.247,  0.130,  0.266,  0.205,  0.302),
            ptMin   = cms.vdouble( 5.000,  5.000,  5.000,  5.000,  5.000,  5.000),
            ptMax   = cms.vdouble(999999, 999999, 999999, 999999, 999999, 999999),
            kind    = cms.string('calo'),
    ),
    resolTrack  = cms.PSet(
            etaBins = cms.vdouble( 0.800,  1.200,  1.500,  2.000,  2.500),
            offset  = cms.vdouble( 0.007,  0.009,  0.011,  0.015,  0.025),
            scale   = cms.vdouble( 0.275,  0.404,  0.512,  0.480,  1.132),
            kind    = cms.string('track'),
    )
 
)

