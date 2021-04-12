import FWCore.ParameterSet.Config as cms

TTStubAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_official_Phase2TrackerDigi_",
    BarrelCut = cms.vdouble(
        0, 2, 2.5, 3.5, 4.5,
        5.5, 7
    ),
    EndcapCutSet = cms.VPSet(
        cms.PSet(
            EndcapCut = cms.vdouble(0)
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 1, 2.5, 2.5, 3,
                2.5, 3, 3.5, 4, 4,
                4.5, 3.5, 4, 4.5, 5,
                5.5
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 0.5, 2.5, 2.5, 3,
                2.5, 3, 3, 3.5, 3.5,
                4, 3.5, 3.5, 4, 4.5,
                5
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 1, 3, 3, 2.5,
                3.5, 3.5, 3.5, 4, 3.5,
                3.5, 4, 4.5
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 1, 2.5, 3, 2.5,
                3.5, 3, 3, 3.5, 3.5,
                3.5, 4, 4
            )
        ),
        cms.PSet(
            EndcapCut = cms.vdouble(
                0, 0.5, 1.5, 3, 2.5,
                3.5, 3, 3, 3.5, 4,
                3.5, 4, 3.5
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
                0, 3, 3, 2.5, 3,
                3, 2.5, 2.5, 2, 1.5,
                1.5, 1, 1
            )
        ),
        cms.PSet(
            TiltedCut = cms.vdouble(
                0, 3.5, 3, 3, 3,
                3, 2.5, 2.5, 3, 3,
                2.5, 2.5, 2.5
            )
        ),
        cms.PSet(
            TiltedCut = cms.vdouble(
                0, 4, 4, 4, 3.5,
                3.5, 3.5, 3.5, 3, 3,
                3, 3, 3
            )
        )
    ),
    zMatching2S = cms.bool(True),
    zMatchingPS = cms.bool(True)
)
