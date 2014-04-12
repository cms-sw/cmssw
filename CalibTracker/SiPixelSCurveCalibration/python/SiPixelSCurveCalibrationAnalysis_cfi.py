import FWCore.ParameterSet.Config as cms

# Put here the modules you want the cfg file to use,
# then include this file in your cfg file.
# i.e. in SiPixelSCurveCalibration.cfg replace 'module demo = SiPixelSCurveCalibration {} '
# with 'include "anlyzerDir/SiPixelSCurveCalibration/data/SiPixelSCurveCalibration.cfi" '.
# (Remember that filenames are case sensitive.)
siPixelSCurveAnalysis = cms.EDFilter("SiPixelSCurveCalibrationAnalysis",
    #setting this value to false will put all folders in one 
    useDetectorHierarchyFolders = cms.untracked.bool(True),
    saveFile = cms.untracked.bool(True),
    # entire DetIDs
    # Watch out, setting this too high on corrupted data will cause HUGE memory consumption!
    detIDsToSave = cms.untracked.vuint32(),
    minimumThreshold = cms.untracked.double(0.0),
    # example  { 352394505, 352394505 }  AGAIN, this will eat A LOT of memory
    #these values are all very large (or small) - user should edit them for what they want to look for
    minimumChi2prob = cms.untracked.double(0.8),
    minimumSigma = cms.untracked.double(0.0),
    write2dFitResult = cms.untracked.bool(True),
    maxCurvesToSave = cms.untracked.uint32(1000), ## limit the maximum number of bad curves to save. This limit applies to both saving bad-flagged error histograms and

    maximumEffAsymptote = cms.untracked.double(1.01), ##this is a pretty silly parameter but it could be helpful for debugging in the future

    write2dHistograms = cms.untracked.bool(True),
    maximumSigma = cms.untracked.double(10.0),
    minimumEffAsymptote = cms.untracked.double(0.0),
    outputFileName = cms.string('Pixel_DQM_Calibration.root'),
    #parameters common to SiPixelOfflineCalibAnalysisBase 
    DetSetVectorSiPixelCalibDigiTag = cms.InputTag("siPixelCalibDigis"),
    maximumThreshold = cms.untracked.double(255.0),
    saveCurvesThatFlaggedBad = cms.untracked.bool(False),
    maximumSigmaBin = cms.untracked.double(10.0),
    maximumThresholdBin = cms.untracked.double(255.0),
    # write out the sigma and thresholds. Needed as input for hardware                             
    writeOutThresholdSummary = cms.untracked.bool(True),
    thresholdOutputFileName = cms.untracked.string("thresholds.txt"),
    alsoWriteZeroThresholds = cms.untracked.bool(False)
)


