import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPPSDiamondSampicTiming')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSDiamondSampicTimingCalib_*_*',
    )
)

OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib = OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib_noDrop.clone()
OutALCARECOPromptCalibProdPPSDiamondSampicTimingCalib.outputCommands.insert(0, 'drop *')
