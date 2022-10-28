import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdPPSAlignment_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPPSAlignment')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertPPSAlignment_*_*',
    )
)

OutALCARECOPromptCalibProdPPSAlignment = OutALCARECOPromptCalibProdPPSAlignment_noDrop.clone()
OutALCARECOPromptCalibProdPPSAlignment.outputCommands.insert(0, 'drop *')
