import FWCore.ParameterSet.Config as cms

# configuration for LaserAlignment
# 
# include the right cff file for the AlignmentAlgorithm which contains
# the corresponding configuration to the used misalignment scenario!
#
# this case: NO misalignment
#
from Alignment.LaserAlignment.BeamProfileFitter_cff import *
from Alignment.LaserAlignment.LaserAlignmentAlgorithm_cff import *
LaserAlignment = cms.EDFilter("LaserAlignment",
    # configuration of the AlignmentAlgorithm (old ones)
    LaserAlignmentAlgorithm,
    # configuration of the BeamProfileFitter
    BeamProfileFitterBlock,
    MinAdcCounts = cms.untracked.int32(0),
    # these are obsolete, to be cleaned in an upcoming fix
    SearchWindowPhiTIB = cms.untracked.double(0.05),
    NumberOfEventsForAllIntensities = cms.untracked.int32(1000),
    UseBrunosAlignmentAlgorithm = cms.untracked.bool(True), ## run the analytical endcap algorithm (exists also as new)

    # create a ROOT file containing the collected profile histograms?
    saveHistograms = cms.untracked.bool(False),
    DoAlignmentAfterNEvents = cms.untracked.int32(25000),
    # enable the new algorithms (and disable the old ones: BrunosAlignmentAlgorithm, Millepede, ...)
    UseNewAlgorithms = cms.untracked.bool(True),
    SearchWindowPhiTOB = cms.untracked.double(0.05),
    UseBeamSplitterFrame = cms.untracked.bool(True),
    # list of digi producers
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('\0'),
        DigiProducer = cms.string('siStripDigis')
    )),
    ROOTFileCompression = cms.untracked.int32(1),
    # if not, here's the steering for the old algorithms
    AlignPosTEC = cms.untracked.bool(False),
    SearchWindowZTOB = cms.untracked.double(1.0),
    # enable the zero (empty profile) filter in the LASProfileJudge, so profiles without signal are rejected.
    # might want to disable this for simulated data with typically low signal level on the last disks
    EnableJudgeZeroFilter = cms.untracked.bool(True),
    # whether to create an sqlite file with a TrackerAlignmentRcd + error
    saveToDbase = cms.untracked.bool(True),
    DebugLevel = cms.untracked.int32(4),
    ROOTFileName = cms.untracked.string('LaserAlignment.histos.root'),
    AlignTECTIBTOBTEC = cms.untracked.bool(False), ## cannot be enabled in this version

    SearchWindowPhiTEC = cms.untracked.double(0.05),
    # do pedestal subtraction. DISABLE THIS for simulated data.
    SubtractPedestals = cms.untracked.bool(True),
    PhiErrorScalingFactor = cms.untracked.double(1.0),
    # currently without effect, to be cleaned in an upcoming fix
    NumberOfEventsPerLaserIntensity = cms.untracked.int32(1000),
    SearchWindowZTIB = cms.untracked.double(1.0),
    AlignNegTEC = cms.untracked.bool(False) ## cannot be enabled in this version

)


