
import FWCore.ParameterSet.Config as cms
process = cms.Process( "Alignment" )

from Alignment.KalmanAlignmentAlgorithm.KalmanAlignmentAlgorithm_Commons import loadKAACommons
loadKAACommons( cms, process )

# message logger
process.MessageLogger = cms.Service( "MessageLogger",

    categories = cms.untracked.vstring( "Alignment", "AlignmentIORootBase" ),

    statistics = cms.untracked.vstring( "alignmentISN" ),
    destinations = cms.untracked.vstring( "alignmentISN" ),

    alignment = cms.untracked.PSet(
        noLineBreaks = cms.untracked.bool( True ),
        threshold = cms.untracked.string( "INFO" ),

        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),

        TrackProducer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        Alignment = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        AlignmentIORootBase = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
    )
)

process.AlignmentProducer.doMisalignmentScenario = cms.bool( False )

process.AlignmentProducer.ParameterBuilder = cms.PSet(
    parameterTypes = cms.vstring( "Selector,RigidBody" ),
    Selector = cms.PSet(
        alignParams = cms.vstring(
            "PixelHalfBarrelLayers,111111", 
            "PXECLayers,111111", 
            "TIBHalfBarrels,111000", 
            "TIDs,111000", 
            "TOBHalfBarrels,111000", 
            "TECs,111000"
        )
    )
)

process.AlignmentProducer.ParameterStore.UseExtendedCorrelations = cms.untracked.bool( False )

process.ReferenceTrajectoryFactory.UseHitWithoutDet = cms.bool( False )

process.AlignmentProducer.algoConfig.AlgorithmConfig = cms.PSet(
    debug = cms.untracked.bool( True ),
    src = cms.string( "" ),
    bsSrc = cms.string( "" ),
    Fitter = cms.string( "KFFittingSmoother" ),
    Propagator = cms.string( "AnalyticalPropagator" ),
    TTRHBuilder = cms.string( "WithoutRefit" ),

    Setups = cms.vstring( "FullTracking" ),

    FullTracking = cms.PSet(
        AlignmentUpdator = cms.PSet( process.SingleTrajectoryUpdatorForPixels ),
        MetricsUpdator = cms.PSet( process.DummyMetricsUpdator ),
        TrajectoryFactory = cms.PSet( process.ReferenceTrajectoryFactory ),

        Tracking = cms.vint32( 1, 2, 3, 4, 5, 6 ),
        External = cms.vint32(),

        PropagationDirection = cms.untracked.string( "alongMomentum" ),
        SortingDirection = cms.untracked.string( "SortInsideOut" ),
        MinTrackingHits = cms.untracked.uint32(13)
    )
)

process.AlignmentProducer.algoConfig.ParameterConfig = cms.PSet(
    ApplyRandomStartValues = cms.untracked.bool( False ),
    UpdateGraphs = cms.untracked.int32(1000),

    InitializationSelector = cms.vstring( "OuterTrackerDets", "InnerTrackerLayers", "InnerTrackerDets", "FixedAlignables", "FreeAlignables" ),

    OuterTrackerDets = cms.PSet(
        AlignableSelection = cms.vstring( "TOBDets", "TECDets" ),

        ApplyParametersFromFile = cms.untracked.bool( True ),
        FileName = cms.untracked.string( "MSSDIR/outer-tracker-dets.root" )
    ),

    InnerTrackerLayers = cms.PSet(
        AlignableSelection = cms.vstring( "TIBLayers", "TIDLayers", "TECLayers", "TOBHalfBarrels" ),

        ApplyParametersFromFile = cms.untracked.bool( True ),
        FileName = cms.untracked.string( "MSSDIR/inner-tracker-layers.root" )
    ),

    InnerTrackerDets = cms.PSet(
        AlignableSelection = cms.vstring( "TIBDets", "TIDDets" ),

        ApplyParametersFromFile = cms.untracked.bool( True ),
        FileName = cms.untracked.string( "MSSDIR/inner-tracker-dets.root" )
    ),

    FixedAlignables = cms.PSet(
        AlignableSelection = cms.vstring( "TIBHalfBarrels", "TIDs", "TOBHalfBarrels", "TECs" ),

        XShiftsStartError = cms.untracked.double(1e-10),
        YShiftsStartError = cms.untracked.double(1e-10),
        ZShiftsStartError = cms.untracked.double(1e-10),
        XRotationsStartError = cms.untracked.double(1e-12),
        YRotationsStartError = cms.untracked.double(1e-12),
        ZRotationsStartError = cms.untracked.double(1e-12)
    ),

    FreeAlignables = cms.PSet(
        AlignableSelection = cms.vstring( "PixelHalfBarrelLayers", "PXECLayers" ),

        XShiftsStartError = cms.untracked.double(0.0002),
        YShiftsStartError = cms.untracked.double(0.0002),
        ZShiftsStartError = cms.untracked.double(0.0002),
        XRotationsStartError = cms.untracked.double(1e-08),
        YRotationsStartError = cms.untracked.double(1e-08),
        ZRotationsStartError = cms.untracked.double(1e-08)
    )
)

process.AlignmentProducer.algoConfig.OutputFile = "kaaOutputISN.root"
process.AlignmentProducer.algoConfig.TimingLogFile = "kaaTimingISN.root"
process.AlignmentProducer.algoConfig.DataCollector.FileName = "kaaDebugISN.root"

process.AlignmentProducer.algoConfig.MergeResults = cms.bool( False )
process.AlignmentProducer.algoConfig.Merger.InputMergeFileNames = cms.vstring()

# track selection
process.AlignmentTrackSelector.src = "ALCARECOTkAlMuonIsolated"
process.AlignmentTrackSelector.ptMin = 5.0

# apply alignment and calibration constants from database
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = "STARTUP_V5::All"

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source( "PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring()
)

