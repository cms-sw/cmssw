import FWCore.ParameterSet.Config as cms

TrackTriggerSetup = cms.ESProducer("trackerDTC::ProducerES",
    DTC = cms.PSet(
        DepthMemory = cms.int32(64),
        NumATCASlots = cms.int32(12),
        NumDTCsPerRegion = cms.int32(24),
        NumModulesPerDTC = cms.int32(72),
        NumOverlappingRegions = cms.int32(2),
        NumRegions = cms.int32(9),
        NumRoutingBlocks = cms.int32(2),
        OffsetDetIdDSV = cms.int32(1),
        OffsetDetIdTP = cms.int32(-1),
        OffsetLayerDisks = cms.int32(10),
        OffsetLayerId = cms.int32(1),
        WidthQoverPt = cms.int32(9),
        WidthRowLUT = cms.int32(4)
    ),
    DuplicateRemoval = cms.PSet(
        DepthMemory = cms.int32(16),
        WidthCot = cms.int32(16),
        WidthPhi0 = cms.int32(12),
        WidthQoverPt = cms.int32(15),
        WidthZ0 = cms.int32(12)
    ),
    Firmware = cms.PSet(
        BField = cms.double(3.81120228767),
        BFieldError = cms.double(1e-06),
        FreqBE = cms.double(360.0),
        FreqLHC = cms.double(40.0),
        HalfLength = cms.double(270.0),
        InnerRadius = cms.double(21.8),
        MaxPitch = cms.double(0.01),
        NumFramesInfra = cms.int32(6),
        OuterRadius = cms.double(112.7),
        SpeedOfLight = cms.double(2.99792458),
        TMP_FE = cms.int32(8),
        TMP_TFP = cms.int32(18)
    ),
    FrontEnd = cms.PSet(
        BaseBend = cms.double(0.25),
        BaseCol = cms.double(1.0),
        BaseRow = cms.double(0.5),
        BaseWindowSize = cms.double(0.5),
        BendCut = cms.double(1.3125),
        WidthBend = cms.int32(6),
        WidthCol = cms.int32(5),
        WidthRow = cms.int32(11)
    ),
    GeometricProcessor = cms.PSet(
        BoundariesEta = cms.vdouble(
            -2.4, -2.08, -1.68, -1.26, -0.9,
            -0.62, -0.41, -0.2, 0.0, 0.2,
            0.41, 0.62, 0.9, 1.26, 1.68,
            2.08, 2.4
        ),
        ChosenRofZ = cms.double(50.0),
        DepthMemory = cms.int32(64),
        NumSectorsPhi = cms.int32(2),
        RangeChiZ = cms.double(90.0)
    ),
    HoughTransform = cms.PSet(
        DepthMemory = cms.int32(32),
        MinLayers = cms.int32(5),
        NumBinsPhiT = cms.int32(32),
        NumBinsQoverPt = cms.int32(16)
    ),
    Hybrid = cms.PSet(
        ChosenRofPhi = cms.double(55.0),
        Disk2SRsSet = cms.VPSet(
            cms.PSet(
                Disk2SRs = cms.vdouble(
                    66.4391, 71.4391, 76.275, 81.275, 82.955,
                    87.955, 93.815, 98.815, 99.816, 104.816
                )
            ),
            cms.PSet(
                Disk2SRs = cms.vdouble(
                    66.4391, 71.4391, 76.275, 81.275, 82.955,
                    87.955, 93.815, 98.815, 99.816, 104.816
                )
            ),
            cms.PSet(
                Disk2SRs = cms.vdouble(
                    63.9903, 68.9903, 74.275, 79.275, 81.9562,
                    86.9562, 92.492, 97.492, 99.816, 104.816
                )
            ),
            cms.PSet(
                Disk2SRs = cms.vdouble(
                    63.9903, 68.9903, 74.275, 79.275, 81.9562,
                    86.9562, 92.492, 97.492, 99.816, 104.816
                )
            ),
            cms.PSet(
                Disk2SRs = cms.vdouble(
                    63.9903, 68.9903, 74.275, 79.275, 81.9562,
                    86.9562, 92.492, 97.492, 99.816, 104.816
                )
            )
        ),
        DiskZs = cms.vdouble(131.1914, 154.9805, 185.332, 221.6016, 265.0195),
        LayerRs = cms.vdouble(
            24.9316, 37.1777, 52.2656, 68.7598, 86.0156,
            108.3105
        ),
        MaxEta = cms.double(2.5),
        MinPt = cms.double(2.0),
        NumLayers = cms.int32(4),
        NumRingsPS = cms.vint32(11, 11, 8, 8, 8),
        RangesAlpha = cms.vdouble(0.0, 0.0, 0.0, 1024.0),
        RangesR = cms.vdouble(7.5, 7.5, 120.0, 0.0),
        RangesZ = cms.vdouble(240.0, 240.0, 7.5, 7.5),
        WidthsAlpha = cms.vint32(0, 0, 0, 4),
        WidthsBend = cms.vint32(3, 4, 3, 4),
        WidthsPhi = cms.vint32(14, 17, 14, 14),
        WidthsR = cms.vint32(7, 7, 12, 7),
        WidthsZ = cms.vint32(12, 8, 7, 7)
    ),
    KalmanFilter = cms.PSet(
        BaseShiftC00 = cms.int32(5),
        BaseShiftC01 = cms.int32(-3),
        BaseShiftC11 = cms.int32(-7),
        BaseShiftC22 = cms.int32(-3),
        BaseShiftC23 = cms.int32(-5),
        BaseShiftC33 = cms.int32(-3),
        BaseShiftChi2 = cms.int32(-5),
        BaseShiftChi20 = cms.int32(-5),
        BaseShiftChi21 = cms.int32(-5),
        BaseShiftInvR00 = cms.int32(-19),
        BaseShiftInvR11 = cms.int32(-21),
        BaseShiftK00 = cms.int32(-9),
        BaseShiftK10 = cms.int32(-15),
        BaseShiftK21 = cms.int32(-13),
        BaseShiftK31 = cms.int32(-14),
        BaseShiftR00 = cms.int32(-2),
        BaseShiftR11 = cms.int32(3),
        BaseShiftS00 = cms.int32(-1),
        BaseShiftS01 = cms.int32(-7),
        BaseShiftS12 = cms.int32(-3),
        BaseShiftS13 = cms.int32(-3),
        BaseShiftr0 = cms.int32(-3),
        BaseShiftr02 = cms.int32(-5),
        BaseShiftr1 = cms.int32(2),
        BaseShiftr12 = cms.int32(5),
        BaseShiftv0 = cms.int32(-2),
        BaseShiftv1 = cms.int32(3),
        MaxLayers = cms.int32(4),
        MaxSkippedLayers = cms.int32(2),
        MaxStubsPerLayer = cms.int32(4),
        MinLayers = cms.int32(4),
        NumTracks = cms.int32(16),
        WidthLutInvPhi = cms.int32(10),
        WidthLutInvZ = cms.int32(10)
    ),
    MiniHoughTransform = cms.PSet(
        MinLayers = cms.int32(5),
        NumBinsPhiT = cms.int32(2),
        NumBinsQoverPt = cms.int32(2),
        NumDLB = cms.int32(2)
    ),
    ProcessHistory = cms.PSet(
        GeometryConfiguration = cms.string('XMLIdealGeometryESSource@'),
        TTStubAlgorithm = cms.string('TTStubAlgorithm_official_Phase2TrackerDigi_@')
    ),
    SeedFilter = cms.PSet(
        BaseDiffZ = cms.int32(4),
        MinLayers = cms.int32(4),
        PowerBaseCot = cms.int32(-6)
    ),
    SupportedGeometry = cms.PSet(
        XMLFile = cms.string('tracker.xml'),
        XMLLabel = cms.string('geomXMLFiles'),
        XMLPath = cms.string('Geometry/TrackerCommonData/data/PhaseII/'),
        XMLVersions = cms.vstring(
            'TiltedTracker613',
            'TiltedTracker613_MB_2019_04'
        )
    ),
    TMTT = cms.PSet(
        ChosenRofPhi = cms.double(67.24),
        MaxEta = cms.double(2.4),
        MinPt = cms.double(3.0),
        NumLayers = cms.int32(7),
        WidthPhi = cms.int32(14),
        WidthR = cms.int32(12),
        WidthZ = cms.int32(14)
    ),
    TrackFinding = cms.PSet(
        BeamWindowZ = cms.double(15.0),
        MatchedLayers = cms.int32(4),
        MatchedLayersPS = cms.int32(0),
        UnMatchedStubs = cms.int32(1),
        UnMatchedStubsPS = cms.int32(0)
    ),
    TrackingParticle = cms.PSet(
        MaxD0 = cms.double(5.0),
        MaxEta = cms.double(2.4),
        MaxVertR = cms.double(1.0),
        MaxVertZ = cms.double(30.0),
        MinLayers = cms.int32(4),
        MinLayersPS = cms.int32(0)
    )
)
