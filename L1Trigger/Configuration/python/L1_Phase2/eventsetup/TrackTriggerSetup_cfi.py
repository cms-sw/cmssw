import FWCore.ParameterSet.Config as cms

TrackTriggerSetup = cms.ESProducer("tt::ProducerSetup",
    CleanTrackBuilder = cms.PSet(
        DepthMemory = cms.int32(16),
        MaxStubs = cms.int32(4),
        MaxTracks = cms.int32(16),
        MinLayers = cms.int32(4),
        NumBinsCot = cms.int32(4),
        NumBinsInv2R = cms.int32(4),
        NumBinsPhiT = cms.int32(4),
        NumBinsZT = cms.int32(4)
    ),
    DTC = cms.PSet(
        DepthMemory = cms.int32(64),
        NumATCASlots = cms.int32(12),
        NumBarrelLayer = cms.int32(6),
        NumBarrelLayerPS = cms.int32(3),
        NumDTCsPerRegion = cms.int32(24),
        NumModulesPerDTC = cms.int32(72),
        NumOverlappingRegions = cms.int32(2),
        NumRegions = cms.int32(9),
        NumRoutingBlocks = cms.int32(2),
        OffsetDetIdDSV = cms.int32(1),
        OffsetDetIdTP = cms.int32(-1),
        OffsetLayerDisks = cms.int32(10),
        OffsetLayerId = cms.int32(1),
        SlotLimit10gbps = cms.int32(3),
        SlotLimitPS = cms.int32(6),
        WidthInv2R = cms.int32(9),
        WidthRowLUT = cms.int32(4)
    ),
    DuplicateRemoval = cms.PSet(
        DepthMemory = cms.int32(16)
    ),
    Firmware = cms.PSet(
        EnableTruncation = cms.bool(True),
        FreqBEHigh = cms.double(360.0),
        FreqBELow = cms.double(240.0),
        FreqLHC = cms.double(40.0),
        NumFramesInfra = cms.int32(6),
        SpeedOfLight = cms.double(2.99792458),
        TMP_FE = cms.int32(8),
        TMP_TFP = cms.int32(18),
        UseHybrid = cms.bool(True),
        WidthAddrBRAM18 = cms.int32(10),
        WidthAddrBRAM36 = cms.int32(9),
        WidthDSPa = cms.int32(27),
        WidthDSPb = cms.int32(18),
        WidthDSPc = cms.int32(48)
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
        ChosenRofZ = cms.double(57.76),
        DepthMemory = cms.int32(32),
        NumBinsPhiT = cms.int32(2),
        NumBinsZT = cms.int32(32),
        PosBarrel = cms.int32(1),
        PosPS = cms.int32(2),
        PosTilted = cms.int32(0),
        WidthModule = cms.int32(3)
    ),
    HoughTransform = cms.PSet(
        DepthMemory = cms.int32(32),
        MinLayers = cms.int32(5),
        NumBinsInv2R = cms.int32(16),
        NumBinsPhiT = cms.int32(32)
    ),
    Hybrid = cms.PSet(
        BarrelHalfLength = cms.double(120.0),
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
        InnerRadius = cms.double(19.6),
        LayerRs = cms.vdouble(
            24.9316, 37.1777, 52.2656, 68.7598, 86.0156,
            108.3105
        ),
        NumLayers = cms.int32(4),
        NumRingsPS = cms.vint32(11, 11, 8, 8, 8),
        RangesAlpha = cms.vdouble(0.0, 0.0, 0.0, 2048.0),
        RangesR = cms.vdouble(7.5, 7.5, 120.0, 0.0),
        RangesZ = cms.vdouble(240.0, 240.0, 7.5, 7.5),
        WidthsAlpha = cms.vint32(0, 0, 0, 4),
        WidthsBend = cms.vint32(3, 4, 3, 4),
        WidthsPhi = cms.vint32(14, 17, 14, 14),
        WidthsR = cms.vint32(7, 7, 12, 7),
        WidthsRTB = cms.vint32(7, 7, 12, 12),
        WidthsZ = cms.vint32(12, 8, 7, 7)
    ),
    KalmanFilter = cms.PSet(
        ApplyNonLinearCorrection = cms.bool(True),
        CutChi2 = cms.double(2.0),
        MaxGaps = cms.int32(4),
        MaxLayers = cms.int32(8),
        MaxSeedingLayer = cms.int32(4),
        MaxTracks = cms.int32(63),
        MinLayers = cms.int32(4),
        MinLayersPS = cms.int32(0),
        MinSeedDeltaR = cms.double(1.6),
        NumSeedStubs = cms.int32(2),
        NumWorker = cms.int32(1),
        RangeFactor = cms.double(3.0),
        ShiftChi20 = cms.int32(-1),
        ShiftChi21 = cms.int32(-5),
        ShiftInitialC00 = cms.int32(0),
        ShiftInitialC11 = cms.int32(0),
        ShiftInitialC22 = cms.int32(0),
        ShiftInitialC33 = cms.int32(0),
        Use5ParameterFit = cms.bool(False),
        UseSimmulation = cms.bool(False),
        UseTTStubParameters = cms.bool(True),
        UseTTStubResiduals = cms.bool(True),
        WidthChi2 = cms.int32(8)
    ),
    ProcessHistory = cms.PSet(
        GeometryConfiguration = cms.string('XMLIdealGeometryESSource@'),
        TTStubAlgorithm = cms.string('TTStubAlgorithm_official_Phase2TrackerDigi_@')
    ),
    TFP = cms.PSet(
        NumChannel = cms.int32(2),
        WidthCot = cms.int32(16),
        WidthInvR = cms.int32(15),
        WidthPhi0 = cms.int32(12),
        WidthZ0 = cms.int32(12)
    ),
    TMTT = cms.PSet(
        WidthPhi = cms.int32(15),
        WidthR = cms.int32(12),
        WidthZ = cms.int32(14)
    ),
    TrackFinding = cms.PSet(
        BeamWindowZ = cms.double(15.0),
        ChosenRofPhi = cms.double(55.0),
        MaxD0 = cms.double(5.0),
        MaxEta = cms.double(2.5),
        MinLayers = cms.int32(4),
        MinPt = cms.double(2.0),
        MinPtCand = cms.double(1.34),
        NumLayers = cms.int32(8)
    ),
    TrackQuality = cms.PSet(
        NumChannel = cms.int32(2)
    ),
    Tracker = cms.PSet(
        BField = cms.double(3.81120228767395),
        BFieldError = cms.double(1e-06),
        HalfLength = cms.double(270.0),
        InnerRadius = cms.double(21.8),
        LimitPSBarrel = cms.double(125.0),
        LimitsPSDiksR = cms.vdouble(66.0, 66.0, 63.0, 63.0, 63.0),
        LimitsPSDiksZ = cms.vdouble(125.0, 150.0, 175.0, 200.0, 250.0),
        LimitsTiltedR = cms.vdouble(30.0, 45.0, 60.0),
        LimitsTiltedZ = cms.vdouble(15.5, 25.0, 34.3),
        OuterRadius = cms.double(112.7),
        PSDiskLimitsR = cms.vdouble(66.4, 66.4, 64.55, 64.55, 64.55),
        PitchCol2S = cms.double(5.025),
        PitchColPS = cms.double(0.1467),
        PitchRow2S = cms.double(0.009),
        PitchRowPS = cms.double(0.01),
        Scattering = cms.double(0.131283),
        TiltApproxIntercept = cms.double(0.507),
        TiltApproxSlope = cms.double(0.884),
        TiltUncertaintyR = cms.double(0.12),
        TiltedLayerLimitsZ = cms.vdouble(
            15.5, 24.9, 34.3, -1.0, -1.0,
            -1.0
        )
    )
)
