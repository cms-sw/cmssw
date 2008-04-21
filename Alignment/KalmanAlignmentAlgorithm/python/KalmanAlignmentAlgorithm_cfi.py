import FWCore.ParameterSet.Config as cms

# Dummy implementation of KalmanAlignmentAlgorithm.cfi.
# See Alignment/KalmanAlignmentAlgorithm/test for configuration examples.
KalmanAlignmentAlgorithm = cms.PSet(
    algoName = cms.string('KalmanAlignmentAlgorithm'),
    ParameterConfig = cms.PSet(

    ),
    OutputFile = cms.string('output.root'),
    WriteAlignmentParameters = cms.untracked.bool(True),
    TrackRefitter = cms.PSet(
        src = cms.string(''),
        bsSrc = cms.string(''),
        Fitter = cms.string('KFFittingSmoother'),
        TTRHBuilder = cms.string('WithoutRefit'),
        AlgorithmName = cms.string('undefAlgorithm'),
        debug = cms.untracked.bool(True),
        Propagator = cms.string('AnalyticalPropagator')
    ),
    DataCollector = cms.PSet(
        XMin = cms.untracked.double(-20.0),
        NBins = cms.untracked.int32(400),
        XMax = cms.untracked.double(20.0),
        FileName = cms.untracked.string('debug.root')
    ),
    AlgorithmConfig = cms.PSet(
        Setups = cms.vstring()
    ),
    TimingLogFile = cms.untracked.string('timing.log')
)

