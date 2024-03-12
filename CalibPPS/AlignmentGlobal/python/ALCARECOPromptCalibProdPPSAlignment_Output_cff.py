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
# foo bar baz
# N27PB3Jxj74T6
# iQj6jK8tzATu0
