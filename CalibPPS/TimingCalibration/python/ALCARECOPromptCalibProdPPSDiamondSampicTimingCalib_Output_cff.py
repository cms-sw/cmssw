import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdPPSDiamondSampic_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPPSDiamondSampicTiming')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSDiamondSampicTimingCalib_*_*',
    )
)

OutALCARECOPromptCalibProdPPSDiamondSampic = OutALCARECOPromptCalibProdPPSDiamondSampic_noDrop.clone()
OutALCARECOPromptCalibProdPPSDiamondSampic.outputCommands.insert(0, 'drop *')
