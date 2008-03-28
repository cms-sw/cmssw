import FWCore.ParameterSet.Config as cms

#
#  SiPixelIsAliveCalibration.cfi
#  CMSSW configuration init file for pixel alive calibrations in CMSSW>=180
#  Original Author:  Freya Blekman
#          Created:  December 6 2007  
#  $Id: SiPixelIsAliveCalibration.cfi,v 1.8 2008/02/22 14:27:46 fblekman Exp $
#
#
siPixelIsAliveCalibration = cms.EDFilter("SiPixelIsAliveCalibration",
    # efficiency when pixel is defined as 'good':
    minEfficiencyForAliveDef = cms.untracked.double(0.8),
    saveFile = cms.untracked.bool(True),
    #parameters common to SiPixelOfflineCalibAnalysisBase 
    DetSetVectorSiPixelCalibDigiTag = cms.InputTag("siPixelCalibDigis"),
    outputFileName = cms.string('Pixel_DQM_Calibration.root')
)


