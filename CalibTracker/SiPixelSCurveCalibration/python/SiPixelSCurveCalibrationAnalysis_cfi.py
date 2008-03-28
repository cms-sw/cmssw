import FWCore.ParameterSet.Config as cms

# Put here the modules you want the cfg file to use,
# then include this file in your cfg file.
# i.e. in SiPixelSCurveCalibration.cfg replace 'module demo = SiPixelSCurveCalibration {} '
# with 'include "anlyzerDir/SiPixelSCurveCalibration/data/SiPixelSCurveCalibration.cfi" '.
# (Remember that filenames are case sensitive.)
siPixelSCurveAnalysis = cms.EDFilter("SiPixelSCurveCalibrationAnalysis",
    #setting this value to false will put all folders in one 
    useDetectorHierarchyFolders = cms.untracked.bool(True),
    plaquettesToSave = cms.untracked.vstring(''),
    #parameters common to SiPixelOfflineCalibAnalysisBase 
    DetSetVectorSiPixelCalibDigiTag = cms.InputTag("siPixelCalibDigis"),
    minimumThreshold = cms.untracked.double(0.0),
    #these values are all very large (or small) - user should edit them for what they want to look for
    minimumChi2prob = cms.untracked.double(0.4),
    saveFile = cms.untracked.bool(True),
    minimumSigma = cms.untracked.double(0.0),
    write2dFitResult = cms.untracked.bool(True),
    maximumEffAsymptote = cms.untracked.double(100.0),
    write2dHistograms = cms.untracked.bool(True),
    maximumThreshold = cms.untracked.double(260.0),
    minimumEffAsymptote = cms.untracked.double(0.0),
    outputFileName = cms.string('Pixel_DQM_Calibration.root'),
    saveCurvesThatFlaggedBad = cms.untracked.bool(False),
    maximumSigma = cms.untracked.double(20.0)
)


