import FWCore.ParameterSet.Config as cms

l1tLayer2Deregionizer = cms.EDProducer("DeregionizerProducer",
    RegionalPuppiCands = cms.InputTag("l1tLayer1","PuppiRegional"),
    linkConfigs = cms.VPSet(
        cms.PSet(
            nLinksPuppi = cms.uint32(6),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(18),
            outputBoard = cms.int32(0),
            outputRegions = cms.vuint32(
                0, 1, 2, 9, 10,
                11, 18, 19, 20, 27,
                28, 29, 36, 37, 38,
                45, 46, 47
            ),
            partition = cms.string('Barrel')
        ),
        cms.PSet(
            nLinksPuppi = cms.uint32(6),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(18),
            outputBoard = cms.int32(1),
            outputRegions = cms.vuint32(
                3, 4, 5, 12, 13,
                14, 21, 22, 23, 30,
                31, 32, 39, 40, 41,
                48, 49, 50
            ),
            partition = cms.string('Barrel')
        ),
        cms.PSet(
            nLinksPuppi = cms.uint32(6),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(18),
            outputBoard = cms.int32(4),
            outputRegions = cms.vuint32(
                6, 7, 8, 15, 16,
                17, 24, 25, 26, 33,
                34, 35, 42, 43, 44,
                51, 52, 53
            ),
            partition = cms.string('Barrel')
        ),
        cms.PSet(
            nLinksPuppi = cms.uint32(3),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(18),
            outputBoard = cms.int32(3),
            outputRegions = cms.vuint32(
                63, 64, 65, 66, 67,
                68, 69, 70, 71
            ),
            partition = cms.string('HGCal')
        ),
        cms.PSet(
            nLinksPuppi = cms.uint32(3),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(18),
            outputBoard = cms.int32(2),
            outputRegions = cms.vuint32(
                54, 55, 56, 57, 58,
                59, 60, 61, 62
            ),
            partition = cms.string('HGCal')
        ),
        cms.PSet(
            nLinksPuppi = cms.uint32(4),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(12),
            outputBoard = cms.int32(5),
            outputRegions = cms.vuint32(
                72, 73, 74, 75, 76,
                77, 78, 79, 80, 81,
                82, 83, 84, 85, 86,
                87, 88, 89
            ),
            partition = cms.string('HGCalNoTk')
        ),
        cms.PSet(
            nLinksPuppi = cms.uint32(3),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(18),
            outputBoard = cms.int32(7),
            outputRegions = cms.vuint32(
                90, 91, 92, 93, 94,
                95, 96, 97, 98
            ),
            partition = cms.string('HF')
        ),
        cms.PSet(
            nLinksPuppi = cms.uint32(3),
            nOutputFramesPerBX = cms.uint32(9),
            nPuppiPerRegion = cms.uint32(18),
            outputBoard = cms.int32(6),
            outputRegions = cms.vuint32(
                99, 100, 101, 102, 103,
                104, 105, 106, 107
            ),
            partition = cms.string('HF')
        )
    ),
    nInputFramesPerBX = cms.uint32(9),
    nPuppiFinalBuffer = cms.uint32(128),
    nPuppiFirstBuffers = cms.uint32(12),
    nPuppiPerClk = cms.uint32(6),
    nPuppiSecondBuffers = cms.uint32(32),
    nPuppiThirdBuffers = cms.uint32(64)
)
