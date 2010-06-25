import FWCore.ParameterSet.Config as cms




OutALCARECOPromptCalibProd_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProd')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*')
)

import copy

OutALCARECOPromptCalibProd=copy.deepcopy(OutALCARECOPromptCalibProd_noDrop)
OutALCARECOPromptCalibProd.outputCommands.insert(0, "drop *")
