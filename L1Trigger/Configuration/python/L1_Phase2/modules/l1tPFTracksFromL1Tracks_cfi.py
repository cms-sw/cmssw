import FWCore.ParameterSet.Config as cms

l1tPFTracksFromL1Tracks = cms.EDProducer("PFTrackProducerFromL1Tracks",
    L1TrackTag = cms.InputTag("l1tTTTracksFromTrackletEmulation","Level1TTTracks"),
    nParam = cms.uint32(4),
    qualityBits = cms.vstring(
        'momentum.perp > 2 && getStubRefs.size >= 4 && chi2Red < 15',
        'momentum.perp > 2 && getStubRefs.size >= 6 && chi2Red < 15 && chi2 < 50',
        'momentum.perp > 5 && getStubRefs.size >= 4'
    ),
    redigitizeTrackWord = cms.bool(True),
    resolCalo = cms.PSet(
        etaBins = cms.vdouble(
            1.3, 1.7, 2.8, 3.2, 4.0,
            5.0
        ),
        kind = cms.string('calo'),
        offset = cms.vdouble(
            2.688, 1.382, 2.096, 1.022, 0.757,
            0.185
        ),
        ptMax = cms.vdouble(
            999999, 999999, 999999, 999999, 999999,
            999999
        ),
        ptMin = cms.vdouble(
            5.0, 5.0, 5.0, 5.0, 5.0,
            5.0
        ),
        scale = cms.vdouble(
            0.154, 0.341, 0.105, 0.255, 0.208,
            0.306
        )
    ),
    resolTrack = cms.PSet(
        etaBins = cms.vdouble(0.8, 1.2, 1.5, 2.0, 2.5),
        kind = cms.string('track'),
        offset = cms.vdouble(0.007, 0.009, 0.011, 0.015, 0.025),
        scale = cms.vdouble(0.275, 0.404, 0.512, 0.48, 1.132)
    )
)
