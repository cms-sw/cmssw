import FWCore.ParameterSet.Config as cms




OutALCARECOPromptCalibProdSiStrip_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStrip')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_MEtoEDMConvertSiStrip_*_*')
)

import copy

OutALCARECOPromptCalibProdSiStrip=copy.deepcopy(OutALCARECOPromptCalibProdSiStrip_noDrop)
OutALCARECOPromptCalibProdSiStrip.outputCommands.insert(0, "drop *")
