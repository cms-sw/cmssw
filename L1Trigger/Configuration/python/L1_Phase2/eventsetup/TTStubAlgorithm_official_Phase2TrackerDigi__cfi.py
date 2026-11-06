import FWCore.ParameterSet.Config as cms

TTStubAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_official_Phase2TrackerDigi_",
    BarrelCut = cms.vdouble(
        0, 2.0, 2.5, 3.5, 4.0,
        5.5, 6.5
    ),
    EndcapCutSet = cms.VPSet(
        cms.PSet(
            EndcapCut = cms.vdouble(0)
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 1.0, 1.5, 1.5, 2.0,
                2.0, 2.5, 2.5, 3.0, 4.0,
                4.0, 2.5, 3.0, 3.5, 4.0,
                5.0
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 0.5, 1.5, 1.5, 2.0,
                2.0, 2.0, 2.5, 2.5, 3.0,
                3.5, 2.0, 2.5, 3.0, 4.0,
                4.0
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 1.5, 2.0, 2.0, 2.0,
                2.0, 2.5, 3.0, 3.5, 2.5,
                2.5, 3.0, 3.5
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 1.0, 1.5, 1.5, 2.0,
                2.0, 2.0, 2.0, 3.0, 2.0,
                2.0, 3.0, 3.0
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 1.0, 1.5, 1.5, 2.0,
                2.0, 2.0, 2.0, 2.5, 3.0,
                2.0, 2.0, 2.5
            )
        )
    ),
    NTiltedRings = cms.vdouble(
        0.0, 12.0, 12.0, 12.0, 0.0,
        0.0, 0.0
    ),
    TiltedBarrelCutSet = cms.VPSet(
        cms.PSet(
            TiltedCut = cms.vdouble(0)
        ),
        cms.PSet(
            TiltedCut = cms.vdouble(
                0, 2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 1.5, 1.5,
                1.5, 1.0, 1.0
            )
        ),
        cms.PSet(
            TiltedCut = cms.vdouble(
                0, 3.0, 3.0, 3.0, 3.0,
                3.0, 2.5, 2.5, 3.0, 3.0,
                2.5, 2.5, 2.5
            )
        ),
        cms.PSet(
            TiltedCut = cms.vdouble(
                0, 4.0, 4.0, 4.0, 3.5,
                3.5, 3.5, 3.0, 3.0, 2.5,
                2.5, 2.5, 2.5
            )
        )
    ),
    zMatching2S = cms.bool(True),
    zMatchingPS = cms.bool(True)
)
