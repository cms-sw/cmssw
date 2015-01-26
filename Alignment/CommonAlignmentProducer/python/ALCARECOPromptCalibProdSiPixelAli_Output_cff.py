import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixelAli_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelAli')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStrip_*_*')
)

import copy

OutALCARECOPromptCalibProdSiPixelAli=copy.deepcopy(OutALCARECOPromptCalibProd_noDrop)
OutALCARECOPromptCalibProdSiPixelAli.outputCommands.insert(0, "drop *")
