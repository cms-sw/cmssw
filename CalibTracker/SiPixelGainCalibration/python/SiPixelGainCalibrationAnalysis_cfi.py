# The following comments couldn't be translated into the new config version:

# this input combined rejects values above suppressZeroAndPlateausInFitFrac of the full range (0-225) from the fit. This effectively rejects plateau values

# fraction of maximum value. if 0.05 the top 0.05 fraction of the gain curve will not be included in the fit 

# option to reject points with one entry or less (so only VCAL points with two or more adc hits are taken into account... Improves fit stability but means big errors in data might be ignored

# minimum chi2/NDOF at which histograms on pixel level are saved. Set to very high or -1 to disable.

# possiblity to save ALL gain curve histograms. Makes program slow and produces HUGE output files. 
# !!!Use with care!!!!

# Database Record name...

import FWCore.ParameterSet.Config as cms

#
#  sipixelgaincalibrationanalysis.cfi
#  CMSSW configuration init file for pixel gain calibrations in CMSSW>=180
#  Original Author:  Freya Blekman
#          Created:  November 15 2007  
#  $Id: SiPixelGainCalibrationAnalysis.cfi,v 1.20 2008/02/22 13:30:57 fblekman Exp $
#
#
siPixelGainCalibrationAnalysis = cms.EDFilter("SiPixelGainCalibrationAnalysis",
    # parameter set from CondTools/SiPixel/ module SiPixelGalibrationService 
    SiPixelGainCalibrationServiceParameters,
    saveFile = cms.untracked.bool(True),
    maxChi2InHist = cms.untracked.double(50.0),
    savePixelLevelHists = cms.untracked.bool(False),
    saveAllHistograms = cms.untracked.bool(False),
    # try to create database. 'true' setting for expert use only.
    writeDatabase = cms.untracked.bool(False),
    record = cms.string('SiPixelGainCalibrationRcd'),
    suppressPointsWithOneEntryOrLess = cms.untracked.bool(True),
    #parameters common to SiPixelOfflineCalibAnalysisBase 
    DetSetVectorSiPixelCalibDigiTag = cms.InputTag("siPixelCalibDigis"),
    outputFileName = cms.string('Pixel_DQM_Calibration.root'),
    suppressZeroAndPlateausInFitFrac = cms.untracked.double(0.05),
    suppressPlateauInFit = cms.untracked.bool(True),
    minChi2NDFforHistSave = cms.untracked.double(25.0),
    minChi2ProbforHistSave = cms.untracked.double(0.001),
    appendDatabaseMode = cms.untracked.bool(False),
    # maxGainInHist fixes the range of the 1D gain summary plots to [0,maxGainInHist]
    maxGainInHist = cms.untracked.double(3.0)
)


