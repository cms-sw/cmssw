import FWCore.ParameterSet.Config as cms

# configuration for LaserAlignment
# 
# include the right cff file for the AlignmentAlgorithm which contains
# the corresponding configuration to the used misalignment scenario!
#
# this case: only rotations as misalignments
#
LaserAlignmentShiftsAndRotations = cms.EDFilter("LaserAlignment",
    MinAdcCounts = cms.untracked.int32(0),
    SearchWindowPhiTIB = cms.untracked.double(0.05),
    NumberOfEventsForAllIntensities = cms.untracked.int32(1000),
    BeamProfileFitter = cms.PSet(
        ScaleHistogramBeforeFit = cms.untracked.bool(True),
        ClearHistogramAfterFit = cms.untracked.bool(True),
        BSAnglesSystematic = cms.untracked.double(0.0007),
        CorrectBeamSplitterKink = cms.untracked.bool(True),
        MinimalSignalHeight = cms.untracked.double(0.0)
    ),
    UseBrunosAlignmentAlgorithm = cms.untracked.bool(False),
    saveHistograms = cms.untracked.bool(False),
    DoAlignmentAfterNEvents = cms.untracked.int32(25000),
    SearchWindowPhiTOB = cms.untracked.double(0.05),
    PhiErrorScalingFactor = cms.untracked.double(1.0),
    # list of digi producers
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('\0'),
        DigiProducer = cms.string('siStripDigis')
    )),
    ROOTFileCompression = cms.untracked.int32(1),
    AlignPosTEC = cms.untracked.bool(False),
    AlignmentAlgorithm = cms.PSet(
        SecondFixedParameterTEC2TEC = cms.untracked.int32(3),
        FirstFixedParameterTEC2TEC = cms.untracked.int32(2),
        FirstFixedParameterNegTEC = cms.untracked.int32(2),
        SecondFixedParameterNegTEC = cms.untracked.int32(3),
        SecondFixedParameterPosTEC = cms.untracked.int32(3),
        FirstFixedParameterPosTEC = cms.untracked.int32(2)
    ),
    SearchWindowZTOB = cms.untracked.double(1.0),
    saveToDbase = cms.untracked.bool(False),
    DebugLevel = cms.untracked.int32(4),
    ROOTFileName = cms.untracked.string('LaserAlignmentShiftsAndRotationsTEC2TEC.histos.root'),
    AlignTECTIBTOBTEC = cms.untracked.bool(True),
    SearchWindowPhiTEC = cms.untracked.double(0.05),
    UseBeamSplitterFrame = cms.untracked.bool(True),
    NumberOfEventsPerLaserIntensity = cms.untracked.int32(1000),
    SearchWindowZTIB = cms.untracked.double(1.0),
    AlignNegTEC = cms.untracked.bool(False)
)


