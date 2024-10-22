import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdPPSTimingCalib_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPPSTimingCalib')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSTimingCalib_*_*',
    )
)

OutALCARECOPromptCalibProdPPSTimingCalib = OutALCARECOPromptCalibProdPPSTimingCalib_noDrop.clone()
OutALCARECOPromptCalibProdPPSTimingCalib.outputCommands.insert(0, 'drop *')
