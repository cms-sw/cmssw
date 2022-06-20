import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixelAliHG_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelAliHG')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_SiPixelAliMillePedeFileConverterHG_*_*')
)

import copy

OutALCARECOPromptCalibProdSiPixelAliHG=copy.deepcopy(OutALCARECOPromptCalibProdSiPixelAliHG_noDrop)
OutALCARECOPromptCalibProdSiPixelAliHG.outputCommands.insert(0, "drop *")
