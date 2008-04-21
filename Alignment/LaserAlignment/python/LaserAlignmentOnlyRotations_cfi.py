import FWCore.ParameterSet.Config as cms

# configuration for LaserAlignment
# 
# include the right cff file for the AlignmentAlgorithm which contains
# the corresponding configuration to the used misalignment scenario!
#
# this case: only rotations as misalignments
#
from Alignment.LaserAlignment.BeamProfileFitter_cff import *
from Alignment.LaserAlignment.LaserAlignmentAlgorithm_cff import *
LaserAlignmentOnlyRotations = cms.EDFilter("LaserAlignment",
    # configuration of the AlignmentAlgorithm
    LaserAlignmentAlgorithm,
    # configuration of the BeamProfileFitter
    BeamProfileFitterBlock,
    MinAdcCounts = cms.untracked.int32(0),
    SearchWindowPhiTIB = cms.untracked.double(0.05),
    NumberOfEventsForAllIntensities = cms.untracked.int32(1000),
    UseBrunosAlignmentAlgorithm = cms.untracked.bool(True),
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
    SearchWindowZTOB = cms.untracked.double(1.0),
    saveToDbase = cms.untracked.bool(False),
    DebugLevel = cms.untracked.int32(4),
    ROOTFileName = cms.untracked.string('LaserAlignmentOnlyRotations.histos.root'),
    AlignTECTIBTOBTEC = cms.untracked.bool(False),
    SearchWindowPhiTEC = cms.untracked.double(0.05),
    UseBeamSplitterFrame = cms.untracked.bool(True),
    NumberOfEventsPerLaserIntensity = cms.untracked.int32(1000),
    SearchWindowZTIB = cms.untracked.double(1.0),
    AlignNegTEC = cms.untracked.bool(False)
)


