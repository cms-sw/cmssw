import FWCore.ParameterSet.Config as cms

OutALCARECOPromptCalibProdSiPixelLorentzAngle_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixelLorentzAngle')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiPixelLorentzAngle_*_*',
    )
)

import copy
OutALCARECOPromptCalibProdSiPixelLorentzAngle=copy.deepcopy(OutALCARECOPromptCalibProdSiPixelLorentzAngle_noDrop)
OutALCARECOPromptCalibProdSiPixelLorentzAngle.outputCommands.insert(0, "drop *")
