import FWCore.ParameterSet.Config as cms


OutALCARECOPromptCalibProdBeamSpotHPLowPU_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdBeamSpotHPLowPU')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducerHPLowPU_*_*')
)

import copy

OutALCARECOPromptCalibProdBeamSpotHPLowPU=copy.deepcopy(OutALCARECOPromptCalibProdBeamSpotHPLowPU_noDrop)
OutALCARECOPromptCalibProdBeamSpotHPLowPU.outputCommands.insert(0, "drop *")
