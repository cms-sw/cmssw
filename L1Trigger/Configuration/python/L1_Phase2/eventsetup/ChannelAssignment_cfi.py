import FWCore.ParameterSet.Config as cms

ChannelAssignment = cms.ESProducer("trklet::ProducerChannelAssignment",
    DR = cms.PSet(
        MinIdenticalStubs = cms.int32(3),
        NumComparisonModules = cms.int32(32)
    ),
    IRChannelsIn = cms.vint32(
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27, 28, 29,
        30, 31, 32, 33, 34,
        35, 36, 37, 38, 39,
        40, 41, 42, 43, 44,
        45, 46, 47
    ),
    SeedTypes = cms.vstring(
        'L1L2',
        'L2L3',
        'L3L4',
        'L5L6',
        'D1D2',
        'D3D4',
        'L1D1',
        'L2D1'
    ),
    SeedTypesProjectionLayers = cms.PSet(
        D1D2 = cms.vint32(1, 2, 13, 14, 15),
        D3D4 = cms.vint32(1, 11, 12, 15),
        L1D1 = cms.vint32(12, 13, 14, 15),
        L1L2 = cms.vint32(
            3, 4, 5, 6, 11,
            12, 13, 14
        ),
        L2D1 = cms.vint32(1, 12, 13, 14),
        L2L3 = cms.vint32(
            1, 4, 5, 6, 11,
            12, 13, 14
        ),
        L3L4 = cms.vint32(
            1, 2, 5, 6, 11,
            12
        ),
        L5L6 = cms.vint32(1, 2, 3, 4)
    ),
    SeedTypesSeedLayers = cms.PSet(
        D1D2 = cms.vint32(11, 12),
        D3D4 = cms.vint32(13, 14),
        L1D1 = cms.vint32(1, 11),
        L1L2 = cms.vint32(1, 2),
        L2D1 = cms.vint32(2, 11),
        L2L3 = cms.vint32(2, 3),
        L3L4 = cms.vint32(3, 4),
        L5L6 = cms.vint32(5, 6)
    ),
    TM = cms.PSet(
        MuxOrder = cms.vstring(
            'L1L2',
            'L2L3',
            'L1D1',
            'D1D2',
            'D3D4',
            'L2D1',
            'L3L4',
            'L5L6'
        ),
        NumLayers = cms.int32(11),
        WidthCot = cms.int32(14),
        WidthStubId = cms.int32(10)
    )
)
