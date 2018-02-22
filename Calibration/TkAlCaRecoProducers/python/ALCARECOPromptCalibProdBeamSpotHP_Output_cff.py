import FWCore.ParameterSet.Config as cms




OutALCARECOPromptCalibProdBeamSpotHP_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdBeamSpotHP')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducerHP_*_*')
)

import copy

OutALCARECOPromptCalibProdBeamSpotHP=copy.deepcopy(OutALCARECOPromptCalibProdBeamSpotHP_noDrop)
OutALCARECOPromptCalibProdBeamSpotHP.outputCommands.insert(0, "drop *")
