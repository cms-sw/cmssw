import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixelAli_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelAli')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_SiPixelAliMillePedeFileConverter_*_*')
)

OutALCARECOPromptCalibProdSiPixelAli=OutALCARECOPromptCalibProdSiPixelAli_noDrop.clone()
OutALCARECOPromptCalibProdSiPixelAli.outputCommands.insert(0, "drop *")
