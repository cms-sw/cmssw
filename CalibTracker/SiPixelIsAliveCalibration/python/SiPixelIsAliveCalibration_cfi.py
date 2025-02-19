import FWCore.ParameterSet.Config as cms

#
#  SiPixelIsAliveCalibration.cfi
#  CMSSW configuration init file for pixel alive calibrations in CMSSW>=180
#  Original Author:  Freya Blekman
#          Created:  December 6 2007  
#  $Id: SiPixelIsAliveCalibration_cfi.py,v 1.2 2008/04/21 00:27:43 rpw Exp $
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


