import FWCore.ParameterSet.Config as cms

# Dummy implementation of KalmanAlignmentAlgorithm.cfi.
# See Alignment/KalmanAlignmentAlgorithm/test for configuration examples.
KalmanAlignmentAlgorithm = cms.PSet(
    algoName = cms.string( "KalmanAlignmentAlgorithm" ),

    AlgorithmConfig = cms.PSet( Setups = cms.vstring() ),
    ParameterConfig = cms.PSet(),

    WriteAlignmentParameters = cms.untracked.bool( True ),
    OutputFile = cms.string( "output.root" ),

    TimingLogFile = cms.untracked.string( "timing.log" ),

    TrackRefitter = cms.PSet(
        src = cms.string( "" ),
        bsSrc = cms.string( "" ),
        Fitter = cms.string( "KFFittingSmoother" ),
        TTRHBuilder = cms.string( "WithoutRefit" ),
        AlgorithmName = cms.string( "undefAlgorithm" ),
        debug = cms.untracked.bool( True ),
        Propagator = cms.string( "AnalyticalPropagator" )
    ),

    DataCollector = cms.PSet(
        XMin = cms.untracked.double(-20.0),
        NBins = cms.untracked.int32(400),
        XMax = cms.untracked.double(20.0),
        FileName = cms.untracked.string( "debug.root" )
    ),

    MergeResults = cms.bool( False ),
    Merger = cms.PSet(
	InputMergeFileNames = cms.vstring(),
	OutputMergeFileName = cms.string( "kaaMerged.root" ),

	ApplyParameters = cms.bool( False ),
	ApplyErrors = cms.bool( False )
    )
)

