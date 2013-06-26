# The following comments couldn't be translated into the new config version:

# this input combined rejects values above suppressZeroAndPlateausInFitFrac of the full range (0-225) from the fit. This effectively rejects plateau values

# fraction of maximum value. if 0.05 the top 0.05 fraction of the gain curve will not be included in the fit 

# option to reject points with one entry or less (so only VCAL points with two or more adc hits are taken into account... Improves fit stability but means big errors in data might be ignored

# minimum chi2/NDOF at which histograms on pixel level are saved. Set to very high or -1 to disable.

# possiblity to save ALL gain curve histograms. Makes program slow and produces HUGE output files. 
# !!!Use with care!!!!

# Database Record name...

import FWCore.ParameterSet.Config as cms

from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *

#
#  sipixelgaincalibrationanalysis.cfi
#  CMSSW configuration init file for pixel gain calibrations in CMSSW>=180
#  Original Author:  Freya Blekman
#          Created:  November 15 2007  
#  $Id: SiPixelGainCalibrationAnalysis_cfi.py,v 1.9 2008/10/15 14:56:35 fblekman Exp $
#
#
siPixelGainCalibrationAnalysis = cms.EDFilter("SiPixelGainCalibrationAnalysis",
    # parameter set from CondTools/SiPixel/ module SiPixelGalibrationService 
    SiPixelGainCalibrationServiceParameters,
    saveFile = cms.untracked.bool(True),
    maxChi2InHist = cms.untracked.double(50.0),
    savePixelLevelHists = cms.untracked.bool(False),
    saveAllHistograms = cms.untracked.bool(False),
    listOfDetIDs = cms.untracked.vuint32(),                                         
    # try to create database. 'true' setting for expert use only.
    writeDatabase = cms.untracked.bool(False),
    record = cms.string('SiPixelGainCalibrationRcd'),
    suppressPointsWithOneEntryOrLess = cms.untracked.bool(True),
    #parameters common to SiPixelOfflineCalibAnalysisBase 
    DetSetVectorSiPixelCalibDigiTag = cms.InputTag("siPixelCalibDigis"),
    outputFileName = cms.string('Pixel_DQM_Calibration.root'),
    suppressZeroAndPlateausInFitFrac = cms.untracked.double(0.2),
    suppressPlateauInFit = cms.untracked.bool(True),
    minChi2NDFforHistSave = cms.untracked.double(25.0),
    minChi2ProbforHistSave = cms.untracked.double(0.001),
    plateauSlopeMax = cms.untracked.double(1.0),
    appendDatabaseMode = cms.untracked.bool(False),
    # the gain is defined as 1/slope of fit.
    # maxGainInHist fixes the range of the 1D gain summary plots to [0,maxGainInHist]]
    maxGainInHist = cms.untracked.double(25.),
    useVCALHIGH = cms.bool(True),
    # conversion factor to go from VCAL_HIGH to VCAL_LOW. Current best estimate: VCAL_HIGH = 7 * VCAL_LOW, which is encoded in the parameter below 
    vcalHighToLowConversionFac = cms.double(7.0),
    # use this mode if you want to analyze S-Curve data with the Gain analysis
    ignoreMode = cms.untracked.bool(False)                                          
)


