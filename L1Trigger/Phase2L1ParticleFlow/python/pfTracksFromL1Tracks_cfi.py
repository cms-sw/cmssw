import FWCore.ParameterSet.Config as cms

pfTracksFromL1Tracks = cms.EDProducer("PFTrackProducerFromL1Tracks",
    L1TrackTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
    nParam = cms.uint32(4),
    resolCalo = cms.PSet(
            etaBins = cms.vdouble( 1.300,  1.700,  2.800,  3.200,  4.000,  5.000),
            offset  = cms.vdouble( 2.688,  1.382,  2.096,  1.022,  0.757,  0.185),
            scale   = cms.vdouble( 0.154,  0.341,  0.105,  0.255,  0.208,  0.306),
            ptMin   = cms.vdouble( 5.000,  5.000,  5.000,  5.000,  5.000,  5.000),
            ptMax   = cms.vdouble(999999, 999999, 999999, 999999, 999999, 999999),
            kind    = cms.string('calo'),
    ),
    resolTrack  = cms.PSet(
            etaBins = cms.vdouble( 0.800,  1.200,  1.500,  2.000,  2.500),
            offset  = cms.vdouble( 0.007,  0.009,  0.011,  0.015,  0.025),
            scale   = cms.vdouble( 0.275,  0.404,  0.512,  0.480,  1.132),
            kind    = cms.string('track'),
    ),
    qualityBits = cms.vstring(
        "momentum.perp > 2 && getStubRefs.size >= 4 && chi2Red < 15",
        "momentum.perp > 2 && getStubRefs.size >= 6 && chi2Red < 15 && chi2 < 50", # historical reasons
        "momentum.perp > 5 && getStubRefs.size >= 4"
    ),
    redigitizeTrackWord = cms.bool(True),
)

