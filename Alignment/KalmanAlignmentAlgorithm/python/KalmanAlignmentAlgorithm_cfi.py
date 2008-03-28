import FWCore.ParameterSet.Config as cms

# Dummy implementation of KalmanAlignmentAlgorithm.cfi.
# See Alignment/KalmanAlignmentAlgorithm/test for configuration examples.
KalmanAlignmentAlgorithm = cms.PSet(
    algoName = cms.string('KalmanAlignmentAlgorithm'),
    OutputFile = cms.string('output.root'),
    Initialization = cms.PSet(

    ),
    WriteAlignmentParameters = cms.untracked.bool(True),
    TrackRefitter = cms.PSet(

    ),
    DataCollector = cms.PSet(
        XMin = cms.untracked.double(-10.0),
        NBins = cms.untracked.int32(400),
        XMax = cms.untracked.double(10.0),
        FileName = cms.untracked.string('debug.root')
    ),
    TimingLogFile = cms.string('timing.log')
)

