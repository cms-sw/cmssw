import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixel_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixel')
    ),
    outputCommands=cms.untracked.vstring(
        'keep *_ALCARECOPromptCalibProdSiPixel_*_*',
        'keep *_siPixelStatusProducer_*_*')
)

import copy

OutALCARECOPromptCalibProdSiPixel=copy.deepcopy(OutALCARECOPromptCalibProdSiPixel_noDrop)
OutALCARECOPromptCalibProdSiPixel.outputCommands.insert(0, "drop *")
