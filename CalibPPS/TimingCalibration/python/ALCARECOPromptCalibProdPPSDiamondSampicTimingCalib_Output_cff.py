import FWCore.ParameterSet.Config as cms

<<<<<<< HEAD
OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPPSDiamondSampicTiming')
=======
OutALCARECOPromptCalibProdPPSDiamondSampic_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPPSDiamondSampic')
>>>>>>> 2b294546c3ee51493450581eb7729a1e5e139fa3
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSDiamondSampicTimingCalib_*_*',
    )
)

<<<<<<< HEAD
OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib = OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib_noDrop.clone()
OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib.outputCommands.insert(0, 'drop *')
=======
OutALCARECOPromptCalibProdPPSDiamondSampic = OutALCARECOPromptCalibProdPPSDiamondSampic_noDrop.clone()
OutALCARECOPromptCalibProdPPSDiamondSampic.outputCommands.insert(0, 'drop *')
>>>>>>> 2b294546c3ee51493450581eb7729a1e5e139fa3
